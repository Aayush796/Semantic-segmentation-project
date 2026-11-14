import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.unet import UNet
from models.unet_dilated import UNetDilated

from src.config import (
    DEVICE,
    IMAGE_SIZE,
    NUM_CLASSES
)


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def measure_latency(
    model,
    device,
    iterations=50,
    warmup=10
):

    model.eval()

    x = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device
    )

    with torch.no_grad():

        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start

    return (
        total_time / iterations
    ) * 1000


def main():

    models = {
        "baseline": UNet(
            num_classes=NUM_CLASSES
        ),
        "dilated": UNetDilated(
            num_classes=NUM_CLASSES
        )
    }

    print(
        f"Device: {DEVICE}"
    )

    print()

    for name, model in models.items():

        model = model.to(DEVICE)

        parameters = count_parameters(
            model
        )

        latency = measure_latency(
            model,
            DEVICE
        )

        print(
            f"{name}:"
        )

        print(
            f"  Parameters: {parameters:,}"
        )

        print(
            f"  Latency: {latency:.3f} ms"
        )

        print()


if __name__ == "__main__":
    main()