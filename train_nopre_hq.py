import argparse
import os
import matplotlib.pyplot as plt

join = os.path.join
import json
from datetime import datetime
import pandas as pd
import numpy as np
import monai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.transforms.functional import resize, to_pil_image
from typing import Any, Iterable, Tuple, List
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

from segment_anything import sam_model_registry
from segment_anything.modeling import TwoWayTransformer, MaskDecoder
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.mask_decoder_hq import MLP


class NpzDataset(Dataset):
    def __init__(self,
                 npz_path,
                 pixel_mean: List[float] = [123.675, 116.28, 103.53],
                 pixel_std: List[float] = [58.395, 57.12, 57.375],
                 device='cuda:0'
                 ):
        self.npz_path = npz_path
        self.npz_files = sorted(os.listdir(self.npz_path))
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        self.device = device

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        img = np.load(join(self.npz_path, self.npz_files[index]))['img']  # (256, 256, 3)
        gt = np.load(join(self.npz_path, self.npz_files[index]))['gt']  # (256, 256)
        # print(img.shape, gt.shape)
        # (256, 256, 3)(256, 256)
        # print(sam_model.image_encoder.img_size, "222222") 1024

        resize_img = self.apply_image(img)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
        # model input: (1, 3, 1024, 1024)
        # print(resize_img_tensor.shape, "resize ")
        input_image = self.preprocess(resize_img_tensor[None, :, :, :]).to(self.device)  # (1, 3, 1024, 1024)
        # print(input_image.shape, "input")
        assert input_image.shape == (1, 3, 1024, 1024), 'input image should be resized to 1024*1024'

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        # print(input_image.shape, "233333")
        return input_image[0], torch.tensor(gt[None, :, :]).long(), torch.tensor(bboxes).float()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # set target_length 1024
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], 1024)
        return np.array(resize(to_pil_image(image), target_size))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class MaskDecoderHQ(MaskDecoder):
    def __init__(self, model_type='vit_b'):
        super().__init__(transformer_dim=256,
                         transformer=TwoWayTransformer(
                             depth=2,
                             embedding_dim=256,
                             mlp_dim=2048,
                             num_heads=8,
                         ),
                         num_multimask_outputs=3,
                         activation=nn.GELU,
                         iou_head_depth=3,
                         iou_head_hidden_dim=256, )
        assert model_type in ["vit_b", "vit_l", "vit_h"]

        checkpoint_dict = {"vit_b": "work_dir/SAM/sam_vit_b_maskdecoder.pth",
                           "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h': "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("HQ Decoder init from SAM MaskDecoder")
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            hq_token_only: bool,
            interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """

        # early-layer ViT feature, after 1st global attention block in ViT
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)

        cloned_vit_features = vit_features.clone().detach()
        cloned_img_embdeddings = image_embeddings.clone().detach()
        hq_features = self.embedding_encoder(cloned_img_embdeddings) + self.compress_vit_feat(cloned_vit_features)

        # hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        # batch_len = len(image_embeddings)
        # masks = []
        # iou_preds = []
        # # image_ps should repeat batch size times
        # image_pe = torch.repeat_interleave(image_pe, batch_len, dim=0)
        # # print(image_embeddings.shape, image_pe.shape, sparse_prompt_embeddings.shape,
        # #       dense_prompt_embeddings.shape, hq_features.shape)
        # for i_batch in range(batch_len):
        #     mask, iou_pred = self.predict_masks(
        #         image_embeddings=image_embeddings[i_batch].unsqueeze(0),
        #         image_pe=image_pe[i_batch].unsqueeze(0),
        #         sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch].unsqueeze(0),
        #         dense_prompt_embeddings=dense_prompt_embeddings[i_batch].unsqueeze(0),
        #         hq_feature=hq_features[i_batch].unsqueeze(0)
        #     )
        #     masks.append(mask)
        #     iou_preds.append(iou_pred)
        # masks = torch.cat(masks, 0)
        # iou_preds = torch.cat(iou_preds, 0)
        # print(image_embeddings.shape, image_pe.shape, sparse_prompt_embeddings.shape,
        #       dense_prompt_embeddings.shape, hq_features.shape)

        masks, iou_preds = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
        )

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

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]

        if hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam

        return masks, iou_preds

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # print(image_embeddings.shape, image_pe.shape, sparse_prompt_embeddings.shape,
        #       dense_prompt_embeddings.shape, hq_feature.shape)

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        # print(output_tokens.shape, "222")
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        #
        # print(output_tokens.shape, sparse_prompt_embeddings.shape, "shape11")
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # torch.Size([2, 6, 256])torch.Size([2, 256])

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # print(src.shape, dense_prompt_embeddings.shape, "5555")
        src = src + dense_prompt_embeddings
        # print(src.shape, "src")([8, 256, 64, 64])
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        # print(src.shape, pos_src.shape, tokens.shape)
        # torch.Size([2, 256, 64, 64]) torch.Size([512, 64, 64]) torch.Size([2, 7, 256])
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def dice_score(preds, targets):
    smooth = 1.0
    assert preds.size() == targets.size()

    iflat = preds.contiguous().view(-1)
    tflat = targets.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    dice = (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice


class TrainMedSam:
    BEST_VAL_LOSS = float("inf")
    BEST_EPOCH = 0

    def __init__(
            self,
            lr: float = 1e-4,
            batch_size: int = 4,
            epochs: int = 50,
            device: str = "cuda:0",
            model_type: str = "vit_b",
            checkpoint: str = "work_dir/SAM/sam_vit_b_01ec64.pth",
            save_path: str = "work_dir/no_npz",
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.sam_checkpoint_dir = checkpoint
        self.model_type = model_type
        self.save_path = save_path

    def __call__(self, train_dataset):
        """Entry method
        prepare `dataset` and `dataloader` objects

        """

        # Define dataloaders
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # get the model
        model = self.get_model()
        model.to(self.device)

        # Train and evaluate model
        self.train(model, train_loader)
        # Evaluate model on test data

        del model
        torch.cuda.empty_cache()

        self.BEST_EPOCH = 0
        self.BEST_VAL_LOSS = float("inf")

        return dice_score

    def get_model(self):
        sam_model = sam_model_registry[self.model_type](
            checkpoint=self.sam_checkpoint_dir
        ).to(self.device)

        return sam_model

    def train(self, model, train_loader: Iterable, logg=True):
        """Train the model"""
        net = MaskDecoderHQ(self.model_type)
        if torch.cuda.is_available():
            net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[self.device])
        net_without_ddp = net.module
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
        lr_scheduler.last_epoch = 0

        net.train()
        _ = net.to(self.device)

        sam_trans = ResizeLongestSide(model.image_encoder.img_size)

        seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )

        model.train()
        best_loss = 1e10
        losses = []
        for epoch in range(self.epochs):
            epoch_losses = 0
            epoch_loss = []
            epoch_dice = []
            progress_bar = tqdm(train_loader, total=len(train_loader))
            for step, (input_image, mask, bbox) in enumerate(progress_bar):
                input_image = input_image.to(self.device)
                mask = mask.to(self.device)

                H, W = mask.shape[-2], mask.shape[-1]
                box = sam_trans.apply_boxes(bbox, (H, W))
                box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

                # Get predictioin mask
                with torch.inference_mode():
                    # print(image.shape, 'img')
                    image_embeddings, interm_embeddings = model.image_encoder(input_image)  # (B,256,64,64)

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )

                # print(image_embeddings.shape, model.prompt_encoder.get_dense_pe().shape, sparse_embeddings.shape,
                #       dense_embeddings.shape)torch.Size([4, 256, 64, 64]) torch.Size([1, 256, 64, 64]) torch.Size([4, 2, 256]) torch.Size([4, 256, 64, 64])
                mask_predictions, _ = net(
                    image_embeddings=image_embeddings.to(self.device),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                    hq_token_only=True,
                    interm_embeddings=interm_embeddings,
                )

                # Calculate loss
                loss = seg_loss(mask_predictions, mask)

                mask_predictions = (mask_predictions > 0.5).float()
                dice = dice_score(mask_predictions, mask)

                epoch_loss.append(loss.detach().item())
                epoch_dice.append(dice.detach().item())

                # empty gradient
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses += loss.item()
                progress_bar.set_description(f"Epoch {epoch + 1}/{self.epochs}")
                progress_bar.set_postfix(
                    loss=np.mean(epoch_loss), dice=np.mean(epoch_dice)
                )
                progress_bar.update()
            # Evaluate every model
            epoch_losses /= step
            losses.append(epoch_losses)
            lr_scheduler.step()
            print(f'EPOCH: {epoch}, Loss: {epoch_losses}')
            # Save the model checkpoint
            filename = 'HQ-model' + str(epoch) + '.pth'
            state = {
                'model_state_dict': model.state_dict(),
                'net_state_dict': net.state_dict()
            }
            torch.save(state, join(self.save_path, filename))

            # Save the best model
            if epoch_losses < best_loss:
                best_loss = epoch_losses
                best_state = {
                    'model_state_dict': model.state_dict(),
                    'net_state_dict': net.state_dict()
                }
                torch.save(best_state, join(self.save_path, 'best.pth'))

            # %% plot loss
            plt.plot(losses)
            plt.title('Dice + Cross Entropy Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # plt.show() # comment this line if you are running on a server
            plt.savefig(join(self.save_path, 'train_loss.png'))
            plt.close()


if __name__ == '__main__':
    # set up parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz_path",
        type=str,
        default="data/testTrain",
        help="the path to original .npz files"
    )
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--task_name', type=str, default='hq_test_4114')
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=50, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=1, help="batch size"
    )
    parser.add_argument("--model_type", default="vit_b", type=str, required=False)
    parser.add_argument(
        "--checkpoint", default="work_dir/SAM/sam_vit_b_01ec64.pth", type=str,
        help="Path to SAM checkpoint"
    )

    args = parser.parse_args()
    train_dataset = NpzDataset(args.npz_path)
    model_save_path = join(args.work_dir, args.task_name)
    os.makedirs(model_save_path, exist_ok=True)

    train = TrainMedSam(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        checkpoint=args.checkpoint,
        save_path=model_save_path,
    )

    train(train_dataset)
