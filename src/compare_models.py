import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet
from models.unet_dilated import UNetDilated


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


baseline = UNet(num_classes=3)
dilated = UNetDilated(num_classes=3)

baseline_params = count_parameters(baseline)
dilated_params = count_parameters(dilated)

print("Model comparison")
print("----------------")

print(f"Baseline U-Net parameters : {baseline_params:,}")
print(f"Dilated U-Net parameters  : {dilated_params:,}")

difference = dilated_params - baseline_params

print(f"Difference                : {difference:,}")