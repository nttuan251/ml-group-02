import torch
from torch import Tensor, nn

from reproduction_code.model.forward.classification_head import (
    ClassificationHead,
)
from reproduction_code.model.forward.prompt_pool import PromptPool
from reproduction_code.model.forward.vit_feature_extractor import ViTFeatureExtractor


class L2PModel(nn.Module):
    """
    The complete pipeline of the forward pass.
    """

    def __init__(
        self,
        num_classes: int,
        pool_size: int,
        prompt_length: int,
        top_n: int,
        use_frequency_penalty: bool,
        embed_dimension: int = 768,
        key_dimension: int = 768,
    ) -> None:
        super().__init__()

        self.feature_extractor = ViTFeatureExtractor()

        self.prompt_pool = PromptPool(
            pool_size=pool_size,
            prompt_length=prompt_length,
            embed_dimension=embed_dimension,
            key_dimension=key_dimension,
            top_n=top_n,
            use_frequency_penalty=use_frequency_penalty,
        )

        self.prompt_token_classifier = ClassificationHead(
            embed_dimension=embed_dimension,
            num_classes=num_classes,
            prompt_length=prompt_length,
            top_n=top_n,
        )

    def classify_and_calculate_surrogate_loss_with_L2P(
        self, X: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Image classification pipeline:
        - Extracts prompts and calculates surrogate loss using image.
        - Embeds image with CLS tokens and positional embeddings.
        - Prepends prompts to the result.
        - Passes the tokens of the previous step through the transformer layer.
        - Extracts the transformed prompt tokens to classify and return logits.

        Args:
            X (Tensor): Input features.

        Returns:
            tuple[Tensor, Tensor]: Logits and surrogate loss.
        """
        prompts, surrogate_loss = (
            self._retrieve_prompts_and_surrogate_loss_from_raw_data(X)
        )

        input_tokens = self._build_input_tokens_with_prompts(X, prompts)

        transformer_output = self.feature_extractor.transform_and_normalize_tokens(
            input_tokens
        )
        logits = (
            self.prompt_token_classifier.extract_transformed_prompt_tokens_and_classify(
                transformer_output
            )
        )

        return logits, surrogate_loss

    def _build_input_tokens_with_prompts(self, X: Tensor, prompts: Tensor) -> Tensor:
        input_tokens_without_prompts = (
            self.feature_extractor.embed_and_build_input_tokens(X)
        )
        input_tokens = self._combine_tokens(prompts, input_tokens_without_prompts)

        return input_tokens

    def _retrieve_prompts_and_surrogate_loss_from_raw_data(
        self, X: Tensor
    ) -> tuple[Tensor, Tensor]:
        queries = self.feature_extractor.pass_through_model_and_extract_cls_tokens(X)
        prompts, surrogate_loss = (
            self.prompt_pool.select_prompts_and_calculate_loss_from_queries(queries)
        )

        return prompts, surrogate_loss

    def _combine_tokens(self, prompts: Tensor, base_tokens: Tensor) -> Tensor:
        return torch.cat((prompts, base_tokens), dim=1)
