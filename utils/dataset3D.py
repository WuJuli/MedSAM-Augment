import os
import pandas as pd
import numpy as np
from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import nibabel as nib
import SimpleITK as sitk
class MedSamDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_col: str,
        mask_col: str,
        image_dir: Any = None,
        mask_dir: str = None,
        image_size: Tuple = (256, 256),
    ):
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_col = image_col
        self.mask_col = mask_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_file = os.path.join(self.image_dir, row[self.image_col]) if self.image_dir else row[self.image_col]
        mask_file = os.path.join(self.mask_dir, row[self.mask_col]) if self.mask_dir else row[self.mask_col]

        if not os.path.exists(image_file):
            raise FileNotFoundError(f"Couldn't find image {image_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Couldn't find mask {mask_file}")

        image_sitk = sitk.ReadImage(image_file)
        image_data = sitk.GetArrayFromImage(image_sitk)
        mask_sitk = sitk.ReadImage(mask_file)
        mask_data = sitk.GetArrayFromImage(mask_sitk)

        # print(image_data.shape, mask_data.shape)
        # (512, 512, 103)(512, 512, 103)

        return self._preprocess(image_data, mask_data)

    def _preprocess(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Threshold mask to binary
        mask = ((mask > 127.0) * 255.0).astype(np.uint8)
        # Convert to tensor
        image = TF.to_tensor(image).float()
        mask = TF.to_tensor(mask).float()
        # Min-max normalize and scale
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        # Resize
        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, antialias=True)

        bbox = self._get_bbox(mask)
        print(image.shape, mask.shape, bbox.shape)

        return image, mask, bbox

    def _get_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        _, y_indices, x_indices = torch.where(mask > 0)

        x_min, y_min = (x_indices.min(), y_indices.min())
        x_max, y_max = (x_indices.max(), y_indices.max())

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return torch.tensor([x_min, y_min, x_max, y_max])


