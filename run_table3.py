"""
TABLE 3 — CIFAR-10 (AlexNet)
Paper Settings:
  - Modified AlexNet Teacher (~1.65M params)
  - AlexNet-Half Student (~723K params)  
  - 40,000 DIs (4,000 per class)
  - β={1.0, 0.1}, τ=20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os

from models.alexnet import AlexNet, AlexNetHalf
from config import config


def get_cifar_loaders():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10('data/', train=True, download=False, transform=transform_train)
    test_data = datasets.CIFAR10('data/', train=False, download=False, transform=transform_test)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    print(f"  CIFAR-10 Train: {len(train_data):,} | Test: {len(test_data):,}")
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            out = model(images)
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def compute_similarity_matrix(model):
    weights = model.get_final_weights()
    sim = F.cosine_similarity(weights.unsqueeze(1), weights.unsqueeze(0), dim=2)
    row_min = sim.min(dim=1, keepdim=True).values
    row_max = sim.max(dim=1, keepdim=True).values
    sim = (sim - row_min) / (row_max - row_min + 1e-8)
    print(f"  Similarity Matrix: {sim.shape}")
    return sim.cpu().numpy()


def sample_dirichlet(alpha, beta, n):
    a = alpha * beta
    a = np.clip(a, 1e-3, None)
    samples = np.random.dirichlet(a, n)
    return torch.tensor(samples, dtype=torch.float32)


_CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
_CLAMP_MIN = ((0.0 - _CIFAR_MEAN) / _CIFAR_STD)
_CLAMP_MAX = ((1.0 - _CIFAR_MEAN) / _CIFAR_STD)

def generate_di_batch(model, targets, device):
    B = targets.shape[0]
    cmin, cmax = _CLAMP_MIN.to(device), _CLAMP_MAX.to(device)
    di = torch.randn(B, 3, 32, 32, device=device)
    with torch.no_grad():
        di = torch.max(torch.min(di, cmax), cmin)
    di.requires_grad_(True)
    targets = targets.to(device)
    optimizer = torch.optim.Adam([di], lr=0.1)
    
    for _ in range(400):
        optimizer.zero_grad()
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            di.data = torch.max(torch.min(di.data, cmax), cmin)
    return di.detach()


class AugDIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        mn, mx = img.min(), img.max()
        img_n = (img - mn) / (mx - mn + 1e-8)
        img_aug = self.transform(img_n)
        img_aug = img_aug * (mx - mn) + mn
        return img_aug, label


if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("  TABLE 3 — CIFAR-10 (AlexNet)")
    print("  40,000 DIs (4,000 per class)")
    print("="*60)
    
    train_loader, test_loader = get_cifar_loaders()
    
    # ── 1. Teacher-CE (AlexNet) ──
    print("\n  --- AlexNet Teacher-CE ---")
    teacher = AlexNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_teacher = 0
    
    if os.path.exists(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth'):
        teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
        teacher.eval()
        best_teacher = evaluate(teacher, test_loader)
        print(f"  Loaded saved CIFAR Teacher model. Acc: {best_teacher:.2f}%")
        epochs_range = []
    else:
        epochs_range = range(1, 101)
        
    for epoch in epochs_range:
        teacher.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"T {epoch}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(teacher(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(teacher, test_loader)
        if acc > best_teacher:
            best_teacher = acc
            torch.save(teacher.state_dict(), f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth')
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: {acc:.2f}% | Best: {best_teacher:.2f}%")
    
    print(f"  Teacher-CE: {best_teacher:.2f}% (Paper: 83.03%)")
    
    # ── 2. Student-CE (AlexNet-Half) ──
    print("\n  --- AlexNet-Half Student-CE ---")
    student = AlexNetHalf().to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_sce = 0
    
    for epoch in range(1, 101):
        student.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"SCE {epoch}"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(student(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_sce:
            best_sce = acc
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: {acc:.2f}%")
    
    print(f"  Student-CE: {best_sce:.2f}% (Paper: 80.04%)")
    
    # ── 3. Student-KD ──
    print("\n  --- AlexNet-Half Student-KD ---")
    teacher = AlexNet().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
    teacher.eval()
    
    student = AlexNetHalf().to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    best_skd = 0
    
    for epoch in range(1, 101):
        student.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"SKD {epoch}"):
            images = images.to(config.DEVICE)
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)
            soft_t = F.softmax(t_logits / 20, dim=1)
            log_s = F.log_softmax(s_logits / 20, dim=1)
            loss = F.kl_div(log_s, soft_t, reduction='batchmean') * (20 ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_skd:
            best_skd = acc
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: {acc:.2f}%")
    
    print(f"  Student-KD: {best_skd:.2f}% (Paper: 80.08%)")
    
    # ── 4. Generate 40,000 DIs ──
    print("\n  --- Generate 40,000 DIs (4,000 per class) ---")
    teacher.eval()
    sim = compute_similarity_matrix(teacher)
    
    all_dis = []
    all_labels = []
    NUM_PER_CLASS = 4000
    
    for cls in range(10):
        alpha = sim[cls]
        class_dis = []
        for beta in [1.0, 0.1]:
            n_per_beta = NUM_PER_CLASS // 2
            targets = sample_dirichlet(alpha, beta, n_per_beta)
            pbar = tqdm(range(0, n_per_beta, 500), desc=f"  C{cls} β={beta}")
            for start in pbar:
                end = min(start + 500, n_per_beta)
                batch_t = targets[start:end]
                di = generate_di_batch(teacher, batch_t, config.DEVICE)
                class_dis.append(di.cpu())
        ct = torch.cat(class_dis, dim=0)
        all_dis.append(ct)
        all_labels.extend([cls] * NUM_PER_CLASS)
        print(f"  ✅ Class {cls}: {ct.shape[0]} DIs")
    
    all_dis = torch.cat(all_dis, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    torch.save({'di_images': all_dis, 'di_labels': all_labels},
               f'{config.CHECKPOINT_DIR}cifar_alexnet_dis.pth')
    print(f"  Saved: {all_dis.shape}")
    
    # ── 5. ZSKD ──
    print("\n  --- ZSKD (40K DIs) ---")
    di_dataset = AugDIDataset(all_dis, all_labels)
    di_loader = DataLoader(di_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    
    student = AlexNetHalf().to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    best_zskd = 0
    
    for epoch in range(1, 501):
        student.train()
        for batch, _ in di_loader:
            batch = batch.to(config.DEVICE)
            with torch.no_grad():
                t_logits = teacher(batch)
            s_logits = student(batch)
            soft_t = F.softmax(t_logits / 20, dim=1)
            log_s = F.log_softmax(s_logits / 20, dim=1)
            loss = F.kl_div(log_s, soft_t, reduction='batchmean') * (20 ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best_zskd:
            best_zskd = acc
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: {acc:.2f}% | Best: {best_zskd:.2f}%")
    
    print(f"  ZSKD: {best_zskd:.2f}% (Paper: 69.56%)")
    
    # ── FINAL TABLE ──
    print(f"\n{'='*60}")
    print(f"  TABLE 3 — CIFAR-10 (AlexNet) RESULTS")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'Ours':>8s}  {'Paper':>8s}")
    print(f"  {'-'*43}")
    print(f"  {'Teacher-CE':<25s} {f'{best_teacher:.2f}%':>8s}  {'83.03%':>8s}")
    print(f"  {'Student-CE':<25s} {f'{best_sce:.2f}%':>8s}  {'80.04%':>8s}")
    print(f"  {'Student-KD':<25s} {f'{best_skd:.2f}%':>8s}  {'80.08%':>8s}")
    print(f"  {'ZSKD (40K DIs)':<25s} {f'{best_zskd:.2f}%':>8s}  {'69.56%':>8s}")
    print(f"{'='*60}")