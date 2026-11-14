import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.unet import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = UNet(num_classes=3).to(device)

x = torch.randn(2, 3, 256, 256).to(device)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)