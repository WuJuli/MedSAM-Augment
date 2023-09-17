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
from typing import Any, Iterable, Tuple, List
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import resize, to_pil_image
from utils.dataset import MedSamDataset


# for 3D dataset
# from utils.dataset3D import MedSamDataset
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
            lr: float = 1e-5,
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

        sam_trans = ResizeLongestSide(model.image_encoder.img_size)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0)

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
                # process image
                # print(input_image.shape, "batched_img")torch.Size([4, 3, 1024, 1024])
                # print(mask.shape, "batched_mask") torch.Size([4, 1, 256, 256])

                input_image = input_image.to(self.device)
                mask = mask.to(self.device)

                H, W = mask.shape[-2], mask.shape[-1]
                box = sam_trans.apply_boxes(bbox, (H, W))
                box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

                for n, value in model.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                # for name, param in model.image_encoder.named_parameters():
                #     if param.requires_grad:
                #         print(name)
                image_embeddings, interm_embeddings = model.image_encoder(input_image)

                # Get predictioin mask
                with torch.inference_mode():
                    # print(image.shape, 'img')
                    # (B,256,64,64)
                    # print(len(interm_embeddings), "checkout ")
                    # print(len(deformable_embeddings), 233333333333)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )
                # print(image_embeddings.shape, model.prompt_encoder.get_dense_pe().shape, sparse_embeddings.shape,
                #       dense_embeddings.shape)
                # ([4, 256, 64, 64])([1, 256, 64, 64])[4, 2, 256][4, 256, 64, 64]
                mask_predictions, _ = model.mask_decoder(
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
                filename = '6sam_model_no_pre' + str(epoch) + '.pth'
                torch.save(
                    model.state_dict(),
                    join(self.save_path, filename)
                )
            # Evaluate every model
            epoch_losses /= step
            losses.append(epoch_losses)
            print(f'EPOCH: {epoch}, Loss: {epoch_losses}')
            # save the model checkpoint
            filename = 'sam_model_no_pre' + str(epoch) + '.pth'
            torch.save(
                model.state_dict(),
                join(self.save_path, filename)
            )
            # save the best model
            if epoch_losses < best_loss:
                best_loss = epoch_losses
                torch.save(
                    model.state_dict(),
                    join(self.save_path, 'best.pth')
                )

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
    parser.add_argument('--task_name', type=str, default='test')
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=50, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=2, help="batch size"
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
