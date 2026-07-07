import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from models.vgg import VGG19, VGG11
from config import config


def get_cifar_test_loader():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    test_data = datasets.CIFAR10(
        'data/', train=False,
        download=False, transform=transform_test
    )
    return DataLoader(
        test_data, batch_size=256,
        shuffle=False, num_workers=0, pin_memory=True
    )


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            out = model(images)
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def kd_vgg11_v2(teacher, student, di_images, di_labels, epochs=300):
    print(f"\n{'='*45}")
    print(f"  VGG-11 ZSKD v2 — NO AUG, ADAM, HIGHER LR")
    print(f"{'='*45}")

    test_loader = get_cifar_test_loader()

    # No augmentation — direct DIs
    di_dataset = TensorDataset(di_images, di_labels)
    di_loader = DataLoader(
        di_dataset, batch_size=256,
        shuffle=True, num_workers=0, pin_memory=True
    )

    student = student.to(config.DEVICE)

    # Adam with higher LR
    optimizer = optim.Adam(
        student.parameters(),
        lr=0.005,                  # 5x higher than before
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    teacher.eval()
    best_acc = 0
    temperature = 20

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0

        for batch, _ in di_loader:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                t_logits = teacher(batch)
            s_logits = student(batch)

            soft_t = F.softmax(t_logits / temperature, dim=1)
            log_soft_s = F.log_softmax(s_logits / temperature, dim=1)
            loss = F.kl_div(
                log_soft_s, soft_t, reduction='batchmean'
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        test_acc = evaluate(student, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"  Epoch [{epoch:3d}/{epochs}] "
              f"Loss: {total_loss/len(di_loader):.4f} | "
              f"Test: {test_acc:.2f}% | "
              f"Best: {best_acc:.2f}%")

    print(f"\n  VGG-11 Best : {best_acc:.2f}%")
    print(f"  Previous    : 39.12%")
    print(f"  Paper       : 74.10%")
    return best_acc


if __name__ == "__main__":
    # Load teacher
    teacher = VGG19().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}VGG19_cifar_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    # Load saved DIs
    di_data = torch.load(
        f'{config.CHECKPOINT_DIR}cifar_dis.pth',
        map_location='cpu'
    )
    di_images = di_data['di_images']
    di_labels = di_data['di_labels']
    print(f"  DIs Loaded: {di_images.shape}")

    student = VGG11()
    acc = kd_vgg11_v2(teacher, student, di_images, di_labels)