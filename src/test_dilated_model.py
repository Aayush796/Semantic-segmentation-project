import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.unet_dilated import UNetDilated


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = UNetDilated(num_classes=3).to(device)

x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)