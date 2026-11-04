from torch import Tensor, nn


class PromptTokenClassifier(nn.Module):
    def __init__(
        self,
        embed_dimension: int,
        num_classes: int,
        prompt_length: int,
        top_n: int,
    ) -> None:
        super().__init__()

        self.num_prompt_tokens = prompt_length * top_n
        self.classifier = nn.Linear(embed_dimension, num_classes)

    def forward(self, transformer_output: Tensor) -> Tensor:
        prompt_tokens = self._extract_prompt_tokens(transformer_output)
        averaged_features = self._average_tokens(prompt_tokens)
        logits = self._classify_features(averaged_features)

        return logits

    def _extract_prompt_tokens(self, transformer_output: Tensor) -> Tensor:
        return transformer_output[:, : self.num_prompt_tokens, :]

    def _average_tokens(self, prompt_tokens: Tensor) -> Tensor:
        return prompt_tokens.mean(dim=1)

    def _classify_features(self, features: Tensor) -> Tensor:
        return self.classifier(features)
