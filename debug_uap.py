import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.lenet import LeNet5
from config import config

# Load existing UAP and dig deeper
teacher = LeNet5().to(config.DEVICE)
teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}teacher_best.pth', map_location=config.DEVICE))
teacher.eval()

# Check MNIST normalized range
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_data = datasets.MNIST(config.DATA_DIR, train=False, download=False, transform=transform)
loader = DataLoader(test_data, batch_size=100, shuffle=False)

batch = next(iter(loader))[0]
print(f"MNIST Range: [{batch.min():.2f}, {batch.max():.2f}]")
print(f"MNIST Mean : {batch.mean():.4f}")
print(f"MNIST Std  : {batch.std():.4f}")

# Test simple random noise UAP
print("\n--- Baseline Tests ---")
for eps in [0.5, 1.0, 1.5, 2.0]:
    noise = torch.randn(1, 1, 32, 32, device=config.DEVICE) * eps
    fooled = 0
    total = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(config.DEVICE)
            orig = teacher(images).argmax(1)
            pert = teacher(images + noise).argmax(1)
            fooled += (orig != pert).sum().item()
            total += images.size(0)
    print(f"Random noise (eps={eps}): {100.*fooled/total:.2f}%")