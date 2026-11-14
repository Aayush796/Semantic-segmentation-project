import torch
from torchvision.datasets import OxfordIIITPet

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nDownloading/loading Oxford-IIIT Pet dataset...")

dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=True
)

print("\nDataset loaded successfully!")
print("Number of images:", len(dataset))

image, mask = dataset[0]

print("Image type:", type(image))
print("Mask type:", type(mask))
print("Image size:", image.size)
print("Mask size:", mask.size)