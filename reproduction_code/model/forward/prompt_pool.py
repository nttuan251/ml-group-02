import torch
from torch import Tensor, nn


class PromptPool(nn.Module):
    """
    Manages a prompt pool and frequency table.
    """

    prompt_frequencies_batch: Tensor
    prompt_normalized_frequencies_task: Tensor

    def __init__(
        self,
        pool_size: int,
        prompt_length: int,
        embed_dimension: int,
        key_dimension: int,
        top_n: int,
        use_frequency_penalty: bool,
    ) -> None:
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.embed_dimension = embed_dimension
        self.key_dimension = key_dimension
        self.top_n = top_n
        self.use_frequency_penalty = use_frequency_penalty

        self.prompts = nn.Parameter(
            torch.empty(pool_size, prompt_length, embed_dimension)
        )
        self.keys = nn.Parameter(torch.empty(pool_size, key_dimension))

        nn.init.uniform_(self.prompts, -1.0, 1.0)
        nn.init.uniform_(self.keys, -1.0, 1.0)

        self.has_frequency_history = False

        if self.use_frequency_penalty:
            self.register_buffer("prompt_frequencies_batch", torch.zeros(pool_size))
            self.register_buffer(
                "prompt_normalized_frequencies_task", torch.zeros(pool_size)
            )

    def select_prompts_and_calculate_loss_from_queries(
        self, queries: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Fetches prompts based on queries (CLS tokens) and calculates surrogate loss in the process.

        Args:
            queries (Tensor): Queries.

        Returns:
            tuple[Tensor, Tensor]: Prompt tokens and surrogate loss.
        """
        with torch.no_grad():
            similarity_matrix = self._calculate_cosine_similarity(queries, self.keys)

            selected_prompt_indices = self._get_top_n_prompt_indices(similarity_matrix)

            if self.training and self.use_frequency_penalty:
                self._update_frequency_after_batch(selected_prompt_indices)

        prompt_tokens = self._extract_prompts(selected_prompt_indices)
        surrogate_loss = self._calculate_surrogate_loss(
            queries, selected_prompt_indices
        )

        return prompt_tokens, surrogate_loss

    def update_frequency_after_task(self) -> None:
        """
        Updates prompt frequency table at the end of a task. The new updated frequency is then used for future prompt selection.
        """
        if not self.use_frequency_penalty:
            raise Exception(
                "Prompt frequency update called when training configured to not use frequency penalty."
            )

        total_prompt_counts: Tensor = self.prompt_frequencies_batch.sum()

        if total_prompt_counts > 0:
            normalized_frequencies = self.prompt_frequencies_batch / total_prompt_counts
            self.prompt_normalized_frequencies_task = normalized_frequencies
            self.has_frequency_history = True

    def _calculate_cosine_similarity(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Calculates a cosine similarity matrix of each query to each prompt key.

        Args:
            queries (Tensor): Queries.
            keys (Tensor): Prompt keys.

        Returns:
            Tensor: Similarity matrix.
        """
        normalized_queries = nn.functional.normalize(queries, p=2, dim=1)
        normalized_keys = nn.functional.normalize(keys, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_queries, normalized_keys.T)

        return similarity_matrix

    def _get_top_n_prompt_indices(self, similarity_matrix: Tensor) -> Tensor:
        """
        Fetches the top most appropriate prompts based on the input similarity matrix. Frequency penalty may be used at this task during training.

        Args:
            similarity_matrix (Tensor): Similarity matrix.

        Returns:
            Tensor: Select prompts' indices.
        """
        cosine_distances = 1.0 - similarity_matrix

        use_penalty = (
            self.training and self.use_frequency_penalty and self.has_frequency_history
        )

        if use_penalty:
            final_distances = cosine_distances * self.prompt_normalized_frequencies_task
        else:
            final_distances = cosine_distances

        _, selected_indices = torch.topk(
            final_distances, k=self.top_n, dim=1, largest=False
        )

        return selected_indices

    def _extract_prompts(self, selected_indices: Tensor) -> Tensor:
        selected_prompts = self.prompts[selected_indices]
        batch_size = selected_indices.shape[0]

        return selected_prompts.view(batch_size, -1, self.embed_dimension)

    def _calculate_surrogate_loss(
        self, queries: Tensor, selected_indices: Tensor
    ) -> Tensor:
        """
        Calculates surrogate loss based on queries and selected prompts.

        Args:
            queries (Tensor): Queries.
            selected_indices (Tensor): Selected prompt indices.

        Returns:
            Tensor: Surrogate loss.
        """
        selected_keys = self.keys[selected_indices]

        queries_expanded = queries.unsqueeze(1)

        normalized_queries = nn.functional.normalize(queries_expanded, p=2, dim=2)
        normalized_keys = nn.functional.normalize(selected_keys, p=2, dim=2)

        cosine_similarities = (normalized_queries * normalized_keys).sum(dim=2)
        cosine_distances = 1.0 - cosine_similarities

        return cosine_distances.sum(dim=1).mean()

    def _update_frequency_after_batch(self, selected_indices: Tensor) -> None:
        """
        Updates prompt frequency table at the end of a batch. The updates at this step only serve to record the frequency, the changes do not take effect until the end of the current task.
        """
        flattened_indices = selected_indices.flatten()
        index_counts = torch.bincount(flattened_indices, minlength=self.pool_size)

        self.prompt_frequencies_batch += index_counts
