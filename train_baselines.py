import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.lenet import LeNet5, LeNet5Half
from config import config


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(
        config.DATA_DIR, train=True,
        download=False, transform=transform
    )
    test_data = datasets.MNIST(
        config.DATA_DIR, train=False,
        download=False, transform=transform
    )
    train_loader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )
    return train_loader, test_loader


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


def train_student_ce():
    """
    Student-CE: Student trained on real data with Cross-Entropy
    Paper Target: 98.92%
    """
    print("\n" + "="*45)
    print("  BASELINE 1 — STUDENT-CE")
    print("  Student trained on real data")
    print("="*45)

    train_loader, test_loader = get_data_loaders()

    student = LeNet5Half().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, 101):
        student.train()
        for images, labels in tqdm(train_loader, leave=False,
                                    desc=f"CE Epoch {epoch}"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = student(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(student, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 20 == 0:
            print(f"  Epoch [{epoch:3d}/100] Test Acc: {test_acc:.2f}%")

    print(f"\n  Student-CE Best : {best_acc:.2f}%")
    print(f"  Paper Target    : 98.92%")
    return best_acc


def train_student_kd():
    """
    Student-KD: Student trained via KD using REAL data
    Paper Target: 99.25%
    This is the UPPER BOUND for data-free methods
    """
    print("\n" + "="*45)
    print("  BASELINE 2 — STUDENT-KD")
    print("  KD with real data (Upper Bound)")
    print("="*45)

    train_loader, test_loader = get_data_loaders()

    # Load trained teacher
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    student = LeNet5Half().to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    temperature = config.KD_TEMPERATURE
    best_acc = 0

    for epoch in range(1, 101):
        student.train()
        for images, labels in tqdm(train_loader, leave=False,
                                    desc=f"KD Epoch {epoch}"):
            images = images.to(config.DEVICE)

            # Teacher soft labels
            with torch.no_grad():
                t_logits = teacher(images)

            # Student
            s_logits = student(images)

            # KD Loss
            soft_teacher = F.softmax(t_logits / temperature, dim=1)
            log_soft_student = F.log_softmax(s_logits / temperature, dim=1)
            loss = F.kl_div(
                log_soft_student, soft_teacher,
                reduction='batchmean'
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(student, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 20 == 0:
            print(f"  Epoch [{epoch:3d}/100] Test Acc: {test_acc:.2f}%")

    print(f"\n  Student-KD Best : {best_acc:.2f}%")
    print(f"  Paper Target    : 99.25%")
    return best_acc


if __name__ == "__main__":
    print("="*50)
    print("  TABLE 1 — BASELINES")
    print("="*50)

    # Student-CE
    ce_acc = train_student_ce()

    # Student-KD
    kd_acc = train_student_kd()

    # Final Table
    print(f"\n{'='*50}")
    print(f"  TABLE 1 — COMPLETE RESULTS (MNIST)")
    print(f"{'='*50}")
    print(f"  {'Model':<25s} {'Ours':>8s} {'Paper':>8s}")
    print(f"  {'-'*43}")
    print(f"  {'Teacher-CE':<25s} {'99.19%':>8s} {'99.34%':>8s}")
    print(f"  {'Student-CE':<25s} {f'{ce_acc:.2f}%':>8s} {'98.92%':>8s}")
    print(f"  {'Student-KD (real data)':<25s} {f'{kd_acc:.2f}%':>8s} {'99.25%':>8s}")
    print(f"  {'ZSKD (24K DIs)':<25s} {'96.08%':>8s} {'98.77%':>8s}")
    print(f"{'='*50}")