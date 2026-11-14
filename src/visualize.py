import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from models.unet_dilated import UNetDilated

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEVICE,
    IMAGE_SIZE,
    NUM_CLASSES,
    QUALITATIVE_DIR,
    SEED,
    VAL_RATIO
)

from src.dataset import (
    PetSegmentationDataset,
    create_train_val_split
)

from src.utils import set_seed


def load_model(
    model_name
):

    if model_name == "baseline":

        model = UNet(
            num_classes=NUM_CLASSES
        )

    else:

        model = UNetDilated(
            num_classes=NUM_CLASSES
        )

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_best.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


def main():

    set_seed(SEED)

    full_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        image_size=IMAGE_SIZE
    )

    _, val_indices = create_train_val_split(
        len(full_dataset),
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    val_dataset = PetSegmentationDataset(
        root=DATA_ROOT,
        split="trainval",
        indices=val_indices,
        image_size=IMAGE_SIZE
    )

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    baseline = load_model(
        "baseline"
    )

    dilated = load_model(
        "dilated"
    )

    os.makedirs(
        QUALITATIVE_DIR,
        exist_ok=True
    )

    for index, (image, mask) in enumerate(loader):

        if index >= 5:
            break

        image_gpu = image.to(DEVICE)

        with torch.no_grad():

            baseline_output = baseline(
                image_gpu
            )

            dilated_output = dilated(
                image_gpu
            )

        baseline_prediction = torch.argmax(
            baseline_output,
            dim=1
        )[0].cpu()

        dilated_prediction = torch.argmax(
            dilated_output,
            dim=1
        )[0].cpu()

        image = image[0].permute(
            1, 2, 0
        )

        mask = mask[0]

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(16, 4)
        )

        axes[0].imshow(image)
        axes[0].set_title("Input")

        axes[1].imshow(mask)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(baseline_prediction)
        axes[2].set_title("Baseline U-Net")

        axes[3].imshow(dilated_prediction)
        axes[3].set_title("Dilated U-Net")

        for axis in axes:
            axis.axis("off")

        plt.tight_layout()

        output_path = os.path.join(
            QUALITATIVE_DIR,
            f"example_{index + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=150
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()