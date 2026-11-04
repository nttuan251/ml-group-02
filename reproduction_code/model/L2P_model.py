import torch
from torch import Tensor, nn

from reproduction_code.model.prompt_pool import PromptPool
from reproduction_code.model.prompt_token_classifier import PromptTokenClassifier
from reproduction_code.model.vit_feature_extractor import ViTFeatureExtractor


class L2PModel(nn.Module):
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

        self.prompt_token_classifier = PromptTokenClassifier(
            embed_dimension=embed_dimension,
            num_classes=num_classes,
            prompt_length=prompt_length,
            top_n=top_n,
        )

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.feature_extractor.extract_cls_feature(X)
        prompts, surrogate_loss = self.prompt_pool.forward(queries)

        base_tokens = self.feature_extractor.embed_and_build_input_tokens(X)
        combined_tokens = self._combine_tokens(prompts, base_tokens)
        transformer_output = self.feature_extractor.transform_and_normalize_tokens(
            combined_tokens
        )
        logits = self.prompt_token_classifier.forward(transformer_output)

        return logits, surrogate_loss

    def _combine_tokens(self, prompts: Tensor, base_tokens: Tensor) -> Tensor:
        return torch.cat((prompts, base_tokens), dim=1)
