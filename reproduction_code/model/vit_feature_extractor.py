from typing import cast

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224") -> None:
        super().__init__()

        self.vit = cast(
            VisionTransformer,
            timm.create_model(model_name, pretrained=True, num_classes=0),
        )

        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

    def extract_cls_feature(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            input_tokens = self.embed_and_build_input_tokens(X)
            output_tokens = self.transform_and_normalize_tokens(input_tokens)

        return output_tokens[:, 0]

    def embed_and_build_input_tokens(self, X: Tensor) -> Tensor:
        patch_tokens = self._get_patch_tokens(X)
        patch_tokens_with_cls = self._prepend_cls_token(patch_tokens)
        input_tokens = self._add_positional_embeddings(patch_tokens_with_cls)

        return input_tokens

    def transform_and_normalize_tokens(self, input_tokens: Tensor) -> Tensor:
        transformed_tokens = self._pass_through_transformer(input_tokens)
        output_tokens = self._normalize_tokens(transformed_tokens)

        return output_tokens

    def _get_patch_tokens(self, X: Tensor) -> Tensor:
        return self.vit.patch_embed(X)

    def _prepend_cls_token(self, token_sequence: Tensor) -> Tensor:
        assert self.vit.cls_token is not None

        batch_size = token_sequence.shape[0]
        cls_token_batch = self.vit.cls_token.expand(batch_size, -1, -1)

        return torch.cat((cls_token_batch, token_sequence), dim=1)

    def _add_positional_embeddings(self, token_sequence: Tensor) -> Tensor:
        assert self.vit.pos_embed is not None
        return token_sequence + self.vit.pos_embed

    def _pass_through_transformer(self, token_sequence: Tensor) -> Tensor:
        for block in self.vit.blocks:
            token_sequence = block(token_sequence)

        return token_sequence

    def _normalize_tokens(self, token_sequence: Tensor) -> Tensor:
        return self.vit.norm(token_sequence)
