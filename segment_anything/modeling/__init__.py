# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# from .detr import DeformableTransformerEncoderLayer
# from .detr import DeformableTransformerEncoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer

# for sam baseline
from .sam import Sam

# from .image_encoder import ImageEncoderViT  # the ori hq image encoder   DETR
# from .image_encoder_ada import ImageEncoderViT  # add MultiScaleAdapter
# from .image_encoder_detr import ImageEncoderViT  # the ori hq image encoder   DETR
from .image_encoder_ada_DETR import ImageEncoderViT  # the ori hq image encoder   DETR

# from .mask_decoder import MaskDecoder      # my version hq mask decoder
# from .mask_decoder_cascade import MaskDecoder  # not with hq, with cascade mask decoder
# from .mask_decoder_cascade import MaskDecoderHQ  # cascade mask decoder HQ
# from .mask_decoder_ori import MaskDecoder  # the original decoder
# from .mask_decoder_detr import MaskDecoder  # using the deformable attention
from .mask_decoder_DETR import MaskDecoder  # using the deformable attention

#from .tiny_vit_sam import TinyViT
