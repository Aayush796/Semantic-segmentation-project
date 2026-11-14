import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet import UNet
from models.unet_dilated import UNetDilated

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEVICE,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    SEED,
    VAL_RATIO
)

from src.dataset import (
    PetSegmentationDataset,
    create_train_val_split
)

from src.utils import set_seed


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model_name):

    set_seed(SEED)

    print(f"Training model: {model_name}")
    print(f"Device: {DEVICE}")

    full_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        image_size=IMAGE_SIZE
    )

    train_indices, val_indices = create_train_val_split(
        len(full_dataset),
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    train_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        indices=train_indices,
        image_size=IMAGE_SIZE
    )

    val_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        indices=val_indices,
        image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model
    if model_name == "baseline":
        model = UNet(
            num_classes=NUM_CLASSES
        )

    elif model_name == "dilated":
        model = UNetDilated(
            num_classes=NUM_CLASSES
        )

    else:
        raise ValueError(
            "Unknown model name"
        )

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0
    )

    criterion = nn.CrossEntropyLoss()

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{model_name}_best.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Saved: {checkpoint_path}"
            )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src/train.py baseline"
        )

        print(
            "python src/train.py dilated"
        )

        sys.exit(1)

    train_model(sys.argv[1])