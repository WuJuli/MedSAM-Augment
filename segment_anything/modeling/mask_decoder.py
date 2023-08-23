# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by HQ-SAM team
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import copy
from typing import List, Tuple, Type

from .common import LayerNorm2d
from deformable_attention import DeformableAttention

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=768, d_ffn=1024,
                 dropout=0.1, activation="relu"):
        super().__init__()

        # self attention
        self.self_attn = DeformableAttention(
            dim=768,  # feature dimensions
            dim_head=64,  # dimension per head
            heads=12,  # attention heads
            dropout=0.,  # dropout
            downsample_factor=4,  # downsample factor (r in paper)
            offset_scale=4,  # scale of offset, maximum offset
            offset_groups=4,  # number of offset groups, should be multiple of heads
            offset_kernel_size=6,  # offset kernel size
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=int(d_model))

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src):
        # self attention
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        for _, layer in enumerate(self.layers):
            output = layer(src)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            vit_dim: int = 768,
    ) -> None:

        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # HQ-SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim)  # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8,
                          3)  # corresponding new MLP layer for HQ-Ouptput-Token
        self.num_mask_tokens = self.num_mask_tokens + 1

        # three conv fusion layers for obtaining HQ-Feature
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

        self.deformable_encoder_layer = DeformableTransformerEncoderLayer()
        self.deformable_encoder = DeformableTransformerEncoder(self.deformable_encoder_layer, 1)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            hq_token_only: bool,
            interm_embeddings: torch.Tensor,
            deformable_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        weights_deformable = [0.7, 0.1, 0.1, 0.1]

        deformable_features = sum(w * emb for w, emb in zip(weights_deformable, deformable_embeddings))

        with torch.inference_mode():
            deformable_features = self.deformable_encoder(deformable_features)
            deformable_features = deformable_features.permute(0, 3, 1, 2)

        cloned_deformable_features = deformable_features.clone().detach()

        hq_features = self.compress_vit_feat(cloned_deformable_features)

        batch_len = len(image_embeddings)
        image_pe = torch.repeat_interleave(image_pe, batch_len, dim=0)
        masks = []
        iou_preds = []

        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch].unsqueeze(0),
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch].unsqueeze(0),
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch].unsqueeze(0),
                hq_features=hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        # Select the correct mask or masks for output
        if multimask_output:
            # mask with highest score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # singale mask output, default
            mask_slice = slice(0, 1)
            iou_preds = iou_preds[:, mask_slice]
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]
        # print(masks_sam.shape, "sam shape") torch.Size([1, 1, 256, 256])
        # print(masks_hq.shape, "hq shape") torch.Size([1, 1, 256, 256])
        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq
        # Prepare output
        return masks, iou_preds

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b, 1, 1, 1)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h,
                                                                                                             w)
        masks_sam_hq = (hyper_in[:, self.num_mask_tokens - 1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h,
                                                                                                               w)
        # print(masks_sam.shape, masks_sam_hq.shape)torch.Size([1, 4, 256, 256]) torch.Size([1, 1, 256, 256])
        masks = torch.cat([masks_sam, masks_sam_hq], dim=1)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
