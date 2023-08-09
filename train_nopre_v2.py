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
import torchvision.transforms.functional as TF
from typing import Any, Iterable
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
# for 2D dataset
from utils.dataset import MedSamDataset


# for 3D dataset
# from utils.dataset3D import MedSamDataset
class NpzDataset(Dataset):
    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.npz_files = sorted(os.listdir(self.npz_path))

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        img = np.load(join(self.npz_path, self.npz_files[index]))['img']  # (256, 256, 3)
        gt = np.load(join(self.npz_path, self.npz_files[index]))['gt']  # (256, 256)
        # print(img.shape, gt.shape)
        # (256, 256, 3)(256, 256)
        # img = (img * 255).astype(np.uint8)
        
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
        return torch.tensor(img).float(), torch.tensor(gt[None, :, :]).long(), torch.tensor(bboxes).float()


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
            model_type: str = "vit_t",
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
            for step, (img, mask, bbox) in enumerate(progress_bar):
                # print(img.shape, "img")
                img = img[0].cpu()  # Move the tensor to CPU before converting to NumPy
                img = np.uint8(img.numpy())  # Convert the image to uint8 data type
                # img = img.to(self.device)
                resize_img = sam_trans.apply_image(img)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(self.device)
                # model input: (1, 3, 1024, 1024)
                # print(resize_img_tensor.shape, "resize ")
                input_image = model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
                # print(input_image.shape, "input")
                assert input_image.shape == (1, 3, model.image_encoder.img_size,
                                             model.image_encoder.img_size), 'input image should be resized to 1024*1024'

                # process image
                # print(input_image.shape,"222222")
                # input_image = input_image.to(self.device)
                mask = mask.to(self.device)

                H, W = mask.shape[-2], mask.shape[-1]
                box = sam_trans.apply_boxes(bbox, (H, W))
                box_tensor = torch.as_tensor(box, dtype=torch.float, device=self.device)

                # Get predictioin mask
                with torch.inference_mode():
                    # print(image.shape, 'img')
                    image_embeddings = model.image_encoder(input_image)  # (B,256,64,64)

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=box_tensor,
                        masks=None,
                    )

                mask_predictions, _ = model.mask_decoder(
                    image_embeddings=image_embeddings.to(
                        self.device
                    ),  # (B, 256, 64, 64)
                    image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
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
                    join(self.save_path, 'model_best.pth')
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
        default="data/Tr_Release_Part1/Tr_Release_Part1",
        help="the path to original .npz files"
    )
    parser.add_argument('--work_dir', type=str, default='./work_dir')
    parser.add_argument('--task_name', type=str, default='train_tiny_on_batch1')
    parser.add_argument(
        "--num_epochs", type=int, required=False, default=5, help="number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=1, help="batch size"
    )
    parser.add_argument("--model_type", default="vit_t", type=str, required=False)
    parser.add_argument(
        "--checkpoint", default="work_dir/tiny_vit_sam/mobile_sam.pt", type=str,
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
