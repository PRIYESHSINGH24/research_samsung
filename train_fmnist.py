import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from models.lenet import LeNet5, LeNet5Half
from config import config
from similarity_matrix import compute_similarity_matrix
from generate_di import sample_dirichlet_vectors, generate_di_batch


# ═══════════════════════════════════
# DATA
# ═══════════════════════════════════

def get_fmnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_data = datasets.FashionMNIST(
        'data/', train=True,
        download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        'data/', train=False,
        download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data, batch_size=256,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=256,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"  FMNIST Train: {len(train_data):,}")
    print(f"  FMNIST Test : {len(test_data):,}")
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


# ═══════════════════════════════════
# STEP 1: TEACHER TRAINING
# ═══════════════════════════════════

def train_fmnist_teacher(epochs=100):
    print("\n" + "="*50)
    print("  STEP 1: FMNIST TEACHER TRAINING")
    print("="*50)

    train_loader, test_loader = get_fmnist_loaders()
    teacher = LeNet5().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        teacher.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        scheduler.step()
        test_acc = evaluate(teacher, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(teacher.state_dict(),
                       f'{config.CHECKPOINT_DIR}fmnist_teacher.pth')

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"Train: {100.*correct/total:.2f}% | Test: {test_acc:.2f}%")

    print(f"\n  FMNIST Teacher Best: {best_acc:.2f}%")
    return teacher, best_acc


# ═══════════════════════════════════
# STEP 2: STUDENT-CE (Baseline)
# ═══════════════════════════════════

def train_fmnist_student_ce(epochs=100):
    print("\n" + "="*50)
    print("  STEP 2: FMNIST STUDENT-CE")
    print("="*50)

    train_loader, test_loader = get_fmnist_loaders()
    student = LeNet5Half().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"CE Epoch {epoch}"):
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

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}%")

    print(f"\n  Student-CE Best: {best_acc:.2f}%")
    return best_acc


# ═══════════════════════════════════
# STEP 3: STUDENT-KD (Real data KD)
# ═══════════════════════════════════

def train_fmnist_student_kd(teacher, epochs=100):
    print("\n" + "="*50)
    print("  STEP 3: FMNIST STUDENT-KD (Real Data)")
    print("="*50)

    train_loader, test_loader = get_fmnist_loaders()
    student = LeNet5Half().to(config.DEVICE)

    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    teacher.eval()
    temperature = 20
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"KD Epoch {epoch}"):
            images = images.to(config.DEVICE)

            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)

            soft_t = F.softmax(t_logits / temperature, dim=1)
            log_soft_s = F.log_softmax(s_logits / temperature, dim=1)
            loss = F.kl_div(
                log_soft_s, soft_t, reduction='batchmean'
            ) * (temperature ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(student, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}%")

    print(f"\n  Student-KD Best: {best_acc:.2f}%")
    return best_acc


# ═══════════════════════════════════
# STEP 4: GENERATE DIs
# ═══════════════════════════════════

def generate_fmnist_dis(teacher, num_per_class=2400):
    print("\n" + "="*50)
    print("  STEP 4: GENERATE FMNIST DIs")
    print("="*50)

    teacher.eval()
    sim_matrix = compute_similarity_matrix(teacher)

    all_images = []
    all_labels = []
    beta_values = [1.0, 0.1]
    n_per_beta = num_per_class // len(beta_values)

    for class_idx in range(10):
        alpha_k = sim_matrix[class_idx]
        class_dis = []

        for beta in beta_values:
            sampled = sample_dirichlet_vectors(alpha_k, beta, n_per_beta)
            pbar = tqdm(
                range(0, n_per_beta, 32),
                desc=f"  Class {class_idx} β={beta}"
            )
            for start in pbar:
                end = min(start + 32, n_per_beta)
                batch_targets = sampled[start:end]
                di_batch = generate_di_batch(
                    teacher, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ Class {class_idx}: {class_tensor.shape[0]} DIs")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    torch.save({
        'di_images': all_images,
        'di_labels': all_labels,
    }, f'{config.CHECKPOINT_DIR}fmnist_dis.pth')

    print(f"  Total: {all_images.shape}")
    print(f"  ✅ Saved!")
    return all_images, all_labels


# ═══════════════════════════════════
# STEP 5: ZSKD (KD on DIs)
# ═══════════════════════════════════

class AugmentedDIDataset(Dataset):
    def __init__(self, di_images, di_labels):
        self.images = di_images
        self.labels = di_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        img_min = image.min()
        img_max = image.max()
        image_norm = (image - img_min) / (img_max - img_min + 1e-8)
        image_aug = self.transform(image_norm)
        image_aug = image_aug * (img_max - img_min) + img_min
        return image_aug, label


def train_fmnist_zskd(teacher, di_images, di_labels, epochs=500):
    print("\n" + "="*50)
    print("  STEP 5: FMNIST ZSKD")
    print("="*50)

    _, test_loader = get_fmnist_loaders()

    di_dataset = AugmentedDIDataset(di_images, di_labels)
    di_loader = DataLoader(
        di_dataset, batch_size=256,
        shuffle=True, num_workers=0, pin_memory=True
    )

    student = LeNet5Half().to(config.DEVICE)
    optimizer = optim.Adam(
        student.parameters(), lr=0.001, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    teacher.eval()
    temperature = 20
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
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

        scheduler.step()
        test_acc = evaluate(student, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  ZSKD Best: {best_acc:.2f}%")
    return best_acc


# ═══════════════════════════════════
# MAIN
# ═══════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  FASHION-MNIST — COMPLETE PIPELINE")
    print("  Table 2 + Baselines")
    print("="*60)

    # Step 1: Teacher
    teacher, teacher_acc = train_fmnist_teacher(epochs=100)

    # Load best teacher
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth',
                   map_location=config.DEVICE)
    )
    teacher.eval()

    # Step 2: Student-CE
    ce_acc = train_fmnist_student_ce(epochs=100)

    # Step 3: Student-KD
    kd_acc = train_fmnist_student_kd(teacher, epochs=100)

    # Step 4: Generate DIs
    dis, labels = generate_fmnist_dis(teacher, num_per_class=2400)

    # Step 5: ZSKD
    zskd_acc = train_fmnist_zskd(teacher, dis, labels, epochs=500)

    # ═══════════════════════════════════
    # TABLE 2 — FMNIST RESULTS
    # ═══════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TABLE 2 — FASHION-MNIST RESULTS")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'Ours':>8s}  {'Paper':>8s}")
    print(f"  {'-'*43}")
    print(f"  {'Teacher-CE':<25s} {f'{teacher_acc:.2f}%':>8s}  {'91.67%':>8s}")
    print(f"  {'Student-CE':<25s} {f'{ce_acc:.2f}%':>8s}  {'89.98%':>8s}")
    print(f"  {'Student-KD':<25s} {f'{kd_acc:.2f}%':>8s}  {'90.85%':>8s}")
    print(f"  {'ZSKD (24K DIs)':<25s} {f'{zskd_acc:.2f}%':>8s}  {'87.48%':>8s}")
    print(f"{'='*60}")