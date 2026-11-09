from typing import cast

import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn


class ViTFeatureExtractor(nn.Module):
    """
    Holds a pretrained Vision Transformer (ViT).
    Main functionalities include different stages of a model forward class, including: embedding tokens, transforming tokens and extracting CLS tokens after both of these processes.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224.orig_in21k") -> None:
        super().__init__()

        self.vit = cast(
            VisionTransformer,
            timm.create_model(model_name, pretrained=True, num_classes=0),
        )

        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

    def pass_through_model_and_extract_cls_tokens(self, X: Tensor) -> Tensor:
        """
        Passes the image through the whole model before extracting the CLS tokens and returning.

        Args:
            X (Tensor): Input features.

        Returns:
            Tensor: CLS tokens.
        """
        with torch.no_grad():
            input_tokens = self.embed_and_build_input_tokens(X)
            output_tokens = self.transform_and_normalize_tokens(input_tokens)

        return output_tokens[:, 0]

    def embed_and_build_input_tokens(self, X: Tensor) -> Tensor:
        """
        Passes the image through the embedding layers, then prepends CLS tokens and adds positional embeddings to the result tokens before returning.

        Args:
            X (Tensor): Input features.

        Returns:
            Tensor: Embedded tokens with prepended CLS and positional embeddings.
        """
        patch_tokens = self._get_patch_tokens(X)
        patch_tokens_with_cls = self._prepend_cls_token(patch_tokens)
        input_tokens = self._add_positional_embeddings(patch_tokens_with_cls)

        return input_tokens

    def transform_and_normalize_tokens(self, input_tokens: Tensor) -> Tensor:
        """
        Passes the embedded tokens with CLS through the transformer layer of the model, before normalizing and returning.

        Args:
            input_tokens (Tensor): Embedded tokens with CLS.

        Returns:
            Tensor: Normalized transformed tokens.
        """
        transformed_tokens = self._pass_through_transformer(input_tokens)
        output_tokens = self._normalize_tokens(transformed_tokens)

        return output_tokens

    def train(self, mode: bool = True) -> "ViTFeatureExtractor":
        """
        Configures the model training mode. Turns on the appropriate mode for all other modules while ensuring the core model is always in evaluation mode.

        Args:
            mode (bool, optional): Turns on training if True.

        Returns:
            ViTFeatureExtractor: The current class instance.
        """
        super().train(mode)
        self.vit.eval()

        return self

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
