import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from models.vgg import VGG19, VGG11
from models.resnet import ResNet18, ResNet18Half
from config import config
from similarity_matrix import compute_similarity_matrix
from generate_di import sample_dirichlet_vectors


# ═══════════════════════════════════
# CIFAR-10 DATA
# ═══════════════════════════════════
def get_cifar_loaders():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    train_data = datasets.CIFAR10(
        'data/', train=True,
        download=False, transform=transform_train
    )
    test_data = datasets.CIFAR10(
        'data/', train=False,
        download=False, transform=transform_test
    )

    train_loader = DataLoader(
        train_data, batch_size=512,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=256,
        shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"CIFAR-10 Train: {len(train_data):,}")
    print(f"CIFAR-10 Test : {len(test_data):,}")
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
# TRAIN TEACHER
# ═══════════════════════════════════
def train_teacher_cifar(model, model_name, epochs=500):
    print(f"\n{'='*45}")
    print(f"  TRAINING {model_name} ON CIFAR-10")
    print(f"{'='*45}")

    train_loader, test_loader = get_cifar_loaders()
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 350, 450], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, leave=False,
                    desc=f"Epoch [{epoch:3d}/{epochs}]")

        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        scheduler.step()
        test_acc = evaluate(model, test_loader)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"Train: {100.*correct/total:.2f}% | "
                  f"Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(
                model.state_dict(),
                f'{config.CHECKPOINT_DIR}{model_name}_cifar_best.pth'
            )

    print(f"\n  {model_name} Best: {best_acc:.2f}%")
    return model, best_acc


# ═══════════════════════════════════
# GENERATE DIs FOR CIFAR
# ═══════════════════════════════════
def generate_di_batch_cifar(model, target_softmax_batch, device):
    B = target_softmax_batch.shape[0]

    di_batch = torch.randn(
        B, 3, 32, 32,
        device=device
    )
    di_batch.requires_grad_(True)

    target = target_softmax_batch.to(device)
    optimizer = torch.optim.Adam([di_batch], lr=0.1)

    for iteration in range(1500):
        optimizer.zero_grad()
        logits = model(di_batch, temperature=20)
        pred_softmax = F.softmax(logits, dim=1)
        loss = -torch.mean(
            torch.sum(target * torch.log(pred_softmax + 1e-8), dim=1)
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            di_batch.clamp_(-2.5, 2.5)

    return di_batch.detach()


def generate_cifar_dis(model, sim_matrix, num_per_class=4000):
    print(f"\n  Generating {num_per_class * 10} CIFAR DIs...")
    model.eval()

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
                di_batch = generate_di_batch_cifar(
                    model, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ Class {class_idx}: {class_tensor.shape[0]} done")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Total: {all_images.shape}")
    return all_images, all_labels


# ═══════════════════════════════════
# KD FOR CIFAR
# ═══════════════════════════════════
class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
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


def kd_cifar(teacher, student, di_images, di_labels, student_name, epochs=500):
    print(f"\n{'='*45}")
    print(f"  ZSKD: {student_name} ON CIFAR-10")
    print(f"{'='*45}")

    _, test_loader = get_cifar_loaders()

    di_dataset = AugmentedDataset(di_images, di_labels)
    di_loader = DataLoader(
        di_dataset, batch_size=256,
        shuffle=True, num_workers=0, pin_memory=True
    )

    student = student.to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    teacher.eval()
    best_acc = 0
    temperature = 20

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
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}%")

    print(f"\n  {student_name} ZSKD: {best_acc:.2f}%")
    return best_acc


# ═══════════════════════════════════
# MAIN — Table 5 (VGG-19 Teacher)
# ═══════════════════════════════════
if __name__ == "__main__":
    print("="*50)
    print("  SECTION 4.1.6 — LARGE ARCHITECTURES")
    print("  Dataset: CIFAR-10")
    print("="*50)

    # ── Step 1: Train VGG-19 Teacher ──
    teacher = VGG19()
    teacher, teacher_acc = train_teacher_cifar(
        teacher, 'VGG19', epochs=500
    )

    # ── Step 2: Similarity Matrix ──
    teacher = VGG19().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}VGG19_cifar_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    sim = compute_similarity_matrix(teacher)

    # ── Step 3: Generate DIs ──
    dis, labels = generate_cifar_dis(teacher, sim, num_per_class=4000)

    torch.save({
        'di_images': dis,
        'di_labels': labels,
    }, f'{config.CHECKPOINT_DIR}cifar_dis.pth')

    # ── Step 4: ZSKD on VGG-11 ──
    student1 = VGG11()
    vgg11_acc = kd_cifar(
        teacher, student1, dis, labels, 'VGG-11'
    )

    # ── Step 5: ZSKD on ResNet-18 ──
    student2 = ResNet18()
    resnet_acc = kd_cifar(
        teacher, student2, dis, labels, 'ResNet-18'
    )

    # ── Results — Table 5 ──
    print(f"\n{'='*50}")
    print(f"  TABLE 5 — VGG-19 TEACHER (CIFAR-10)")
    print(f"{'='*50}")
    print(f"  {'Model':<25s} {'Ours':>8s} {'Paper':>8s}")
    print(f"  {'-'*43}")
    print(f"  {'VGG-19 (Teacher)':<25s} {f'{teacher_acc:.2f}%':>8s} {'87.99%':>8s}")
    print(f"  {'VGG-11 ZSKD':<25s} {f'{vgg11_acc:.2f}%':>8s} {'74.10%':>8s}")
    print(f"  {'ResNet-18 ZSKD':<25s} {f'{resnet_acc:.2f}%':>8s} {'74.76%':>8s}")
    print(f"{'='*50}")