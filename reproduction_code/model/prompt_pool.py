import torch
from torch import Tensor, nn


class PromptPool(nn.Module):
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

        self.prompts = nn.Embedding(
            pool_size,
            prompt_length * embed_dimension,
            sparse=True,
        )
        self.keys = nn.Embedding(pool_size, key_dimension, sparse=True)

        nn.init.uniform_(self.prompts.weight, -1.0, 1.0)
        nn.init.uniform_(self.keys.weight, -1.0, 1.0)

        self.has_frequency_history = False

        if self.use_frequency_penalty:
            self.register_buffer("prompt_frequencies_batch", torch.zeros(pool_size))
            self.register_buffer(
                "prompt_normalized_frequencies_task", torch.zeros(pool_size)
            )

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            similarity_matrix = self._calculate_cosine_similarity(
                queries, self.keys.weight
            )

            selected_prompt_indices = self._get_top_n_prompt_indices(similarity_matrix)

            if self.training and self.use_frequency_penalty:
                self._update_frequency_batch(selected_prompt_indices)

        prompt_tokens = self._extract_prompts(selected_prompt_indices)
        surrogate_loss = self._calculate_surrogate_loss(
            queries, selected_prompt_indices
        )

        return prompt_tokens, surrogate_loss

    def update_frequency_task(self) -> None:
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
        normalized_queries = nn.functional.normalize(queries, p=2, dim=1)
        normalized_keys = nn.functional.normalize(keys, p=2, dim=1)
        similarity_matrix = torch.matmul(normalized_queries, normalized_keys.T)

        return similarity_matrix

    def _get_top_n_prompt_indices(self, similarity_matrix: Tensor) -> Tensor:
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
        selected_prompts: Tensor = self.prompts(selected_indices)
        batch_size = selected_indices.shape[0]

        return selected_prompts.view(batch_size, -1, self.embed_dimension)

    def _calculate_surrogate_loss(
        self, queries: Tensor, selected_indices: Tensor
    ) -> Tensor:
        selected_keys: Tensor = self.keys(selected_indices)

        queries_expanded = queries.unsqueeze(1)

        normalized_queries = nn.functional.normalize(queries_expanded, p=2, dim=2)
        normalized_keys = nn.functional.normalize(selected_keys, p=2, dim=2)

        cosine_similarities = (normalized_queries * normalized_keys).sum(dim=2)
        cosine_distances = 1.0 - cosine_similarities

        return cosine_distances.sum(dim=1).mean()

    def _update_frequency_batch(self, selected_indices: Tensor) -> None:
        flattened_indices = selected_indices.flatten()
        index_counts = torch.bincount(flattened_indices, minlength=self.pool_size)

        self.prompt_frequencies_batch += index_counts
