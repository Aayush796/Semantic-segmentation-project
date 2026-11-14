import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    METRICS_DIR,
    NUM_CLASSES,
    SEED,
    VAL_RATIO
)

from src.dataset import (
    PetSegmentationDataset,
    create_train_val_split
)

from src.metrics import (
    intersection_over_union,
    mean_iou
)

from src.utils import set_seed


def evaluate_model(model_name):

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

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    if model_name == "baseline":
        model = UNet(num_classes=NUM_CLASSES)

    elif model_name == "dilated":
        model = UNetDilated(num_classes=NUM_CLASSES)

    else:
        raise ValueError(
            "Unknown model name"
        )

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{model_name}_best.pth"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    all_ious = []

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            batch_ious = intersection_over_union(
                predictions,
                masks,
                NUM_CLASSES
            )

            all_ious.append(batch_ious)

    class_ious = []

    for class_id in range(NUM_CLASSES):

        values = [
            batch[class_id]
            for batch in all_ious
            if batch[class_id] == batch[class_id]
        ]

        class_ious.append(
            sum(values) / len(values)
        )

    miou = mean_iou(class_ious)

    os.makedirs(
        METRICS_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        METRICS_DIR,
        f"{model_name}_metrics.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "class",
            "IoU"
        ])

        writer.writerow([
            "background",
            class_ious[0]
        ])

        writer.writerow([
            "pet",
            class_ious[1]
        ])

        writer.writerow([
            "boundary",
            class_ious[2]
        ])

        writer.writerow([
            "mIoU",
            miou
        ])

    print(
        f"\n{model_name} results"
    )

    print(
        f"Background IoU: {class_ious[0]:.4f}"
    )

    print(
        f"Pet IoU       : {class_ious[1]:.4f}"
    )

    print(
        f"Boundary IoU  : {class_ious[2]:.4f}"
    )

    print(
        f"mIoU          : {miou:.4f}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python src/evaluate.py baseline"
        )

        print(
            "python src/evaluate.py dilated"
        )

        sys.exit(1)

    evaluate_model(sys.argv[1])