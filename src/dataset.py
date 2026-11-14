import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PetSegmentationDataset(Dataset):

    def __init__(
        self,
        root="./data",
        split="trainval",
        indices=None,
        image_size=256
    ):
        self.image_size = image_size

        self.dataset = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=False
        )

        if indices is None:
            self.indices = list(range(len(self.dataset)))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        image, mask = self.dataset[real_idx]

        # Resize image using bilinear interpolation
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )

        # Resize segmentation mask using nearest-neighbor
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        # Convert image to tensor and normalize to [0, 1]
        image = TF.to_tensor(image)

        # Convert mask to tensor
        mask = torch.as_tensor(
            np.array(mask),
            dtype=torch.long
        )


        mask = mask - 1

        return image, mask


def create_train_val_split(
    dataset_size,
    val_ratio=0.2,
    seed=42
):

    indices = list(range(dataset_size))

    random.Random(seed).shuffle(indices)

    val_size = int(dataset_size * val_ratio)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return train_indices, val_indices