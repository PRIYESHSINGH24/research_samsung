import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from models.vgg import VGG19, VGG11
from models.resnet import ResNet18, ResNet18Half
from config import config


def get_cifar_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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
        train_data, batch_size=256,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=256,
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


def train_student_ce(model, model_name, epochs=200):
    print(f"\n{'='*45}")
    print(f"  CIFAR STUDENT-CE: {model_name}")
    print(f"{'='*45}")

    train_loader, test_loader = get_cifar_loaders()
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"CE {epoch}"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  {model_name} CE Best: {best_acc:.2f}%")
    return best_acc


def train_student_kd(teacher, student, model_name, epochs=200):
    print(f"\n{'='*45}")
    print(f"  CIFAR STUDENT-KD: {model_name}")
    print(f"{'='*45}")

    train_loader, test_loader = get_cifar_loaders()
    student = student.to(config.DEVICE)
    teacher.eval()

    optimizer = optim.Adam(
        student.parameters(), lr=0.001,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    temperature = 20
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"KD {epoch}"):
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
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  {model_name} KD Best: {best_acc:.2f}%")
    return best_acc


def train_resnet_teacher(epochs=200):
    print(f"\n{'='*45}")
    print(f"  CIFAR RESNET-18 TEACHER")
    print(f"{'='*45}")

    train_loader, test_loader = get_cifar_loaders()
    model = ResNet18().to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       f'{config.CHECKPOINT_DIR}ResNet18_cifar_teacher.pth')

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  ResNet-18 Teacher Best: {best_acc:.2f}%")
    return model, best_acc


def compute_similarity_matrix_resnet(model):
    weights = model.get_final_weights()
    sim = F.cosine_similarity(
        weights.unsqueeze(1), weights.unsqueeze(0), dim=2
    )
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    print(f"  Similarity Matrix: {sim.shape}")
    return sim.cpu().numpy()


def resnet_zskd(teacher, epochs=300):
    print(f"\n{'='*45}")
    print(f"  RESNET ZSKD: Generate DIs + KD")
    print(f"{'='*45}")

    teacher.eval()
    sim_matrix = compute_similarity_matrix_resnet(teacher)

    print("  Generating DIs from ResNet-18...")
    all_images = []
    all_labels = []

    for class_idx in range(10):
        alpha_k = sim_matrix[class_idx]
        class_dis = []

        for beta in [1.0, 0.1]:
            n_samples = 2000
            alpha = alpha_k * beta
            alpha = np.clip(alpha, 1e-3, None)
            targets = np.random.dirichlet(alpha, n_samples)
            targets = torch.tensor(targets, dtype=torch.float32)

            pbar = tqdm(
                range(0, n_samples, 32),
                desc=f"  Class {class_idx} β={beta}"
            )
            for start in pbar:
                end = min(start + 32, n_samples)
                batch_targets = targets[start:end].to(config.DEVICE)
                B = batch_targets.shape[0]

                di_batch = torch.randn(B, 3, 32, 32, device=config.DEVICE)
                di_batch.requires_grad_(True)
                opt = torch.optim.Adam([di_batch], lr=0.1)

                for it in range(1500):
                    opt.zero_grad()
                    logits = teacher(di_batch, temperature=20)
                    pred = F.softmax(logits, dim=1)
                    loss = -torch.mean(
                        torch.sum(batch_targets * torch.log(pred + 1e-8), dim=1)
                    )
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        di_batch.clamp_(-2.5, 2.5)

                class_dis.append(di_batch.detach().cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * 4000)
        print(f"  ✅ Class {class_idx}: {class_tensor.shape[0]} DIs")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    print("\n  Training ResNet-18-Half via KD...")
    _, test_loader = get_cifar_loaders()

    di_dataset = torch.utils.data.TensorDataset(all_images, all_labels)
    di_loader = DataLoader(di_dataset, batch_size=256, shuffle=True, num_workers=0)

    student = ResNet18Half().to(config.DEVICE)
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
            loss = F.kl_div(log_soft_s, soft_t, reduction='batchmean') * (temperature ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(student, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  ResNet-18-Half ZSKD Best: {best_acc:.2f}%")
    return best_acc


if __name__ == "__main__":
    print("="*60)
    print("  CIFAR-10 BASELINES + TABLE 6")
    print("="*60)

    # Load VGG-19 Teacher
    teacher_vgg = VGG19().to(config.DEVICE)
    teacher_vgg.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}VGG19_cifar_best.pth',
                   map_location=config.DEVICE)
    )
    teacher_vgg.eval()

    # ── TABLE 5 BASELINES ──
    print("\n" + "="*60)
    print("  TABLE 5 — VGG BASELINES")
    print("="*60)

    # VGG-11 Student-CE — ALREADY DONE
    vgg11_ce = 92.91
    print(f"\n  VGG-11 CE (previous run): {vgg11_ce}%")

    # VGG-11 Student-KD (FIXED: Adam optimizer)
    vgg11_kd = train_student_kd(teacher_vgg, VGG11(), 'VGG-11', epochs=200)

    # ResNet-18 Student-CE
    resnet_ce = train_student_ce(ResNet18(), 'ResNet-18', epochs=200)

    # ResNet-18 Student-KD
    resnet_kd = train_student_kd(teacher_vgg, ResNet18(), 'ResNet-18', epochs=200)

    print(f"\n{'='*60}")
    print(f"  TABLE 5 — COMPLETE (VGG-19 Teacher)")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'Ours':>8s}  {'Paper':>8s}")
    print(f"  {'-'*43}")
    print(f"  {'VGG-19 Teacher':<25s} {'88.47%':>8s}  {'87.99%':>8s}")
    print(f"  {'VGG-11 Student-CE':<25s} {f'{vgg11_ce:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'VGG-11 Student-KD':<25s} {f'{vgg11_kd:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'VGG-11 ZSKD':<25s} {'39.12%':>8s}  {'74.10%':>8s}")
    print(f"  {'ResNet-18 Student-CE':<25s} {f'{resnet_ce:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'ResNet-18 Student-KD':<25s} {f'{resnet_kd:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'ResNet-18 ZSKD':<25s} {'64.65%':>8s}  {'74.76%':>8s}")
    print(f"{'='*60}")

    # ── TABLE 6 ──
    print("\n" + "="*60)
    print("  TABLE 6 — ResNet-18 TEACHER")
    print("="*60)

    teacher_resnet, resnet_teacher_acc = train_resnet_teacher(epochs=200)

    teacher_resnet = ResNet18().to(config.DEVICE)
    teacher_resnet.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}ResNet18_cifar_teacher.pth',
                   map_location=config.DEVICE)
    )
    teacher_resnet.eval()

    resnet_half_ce = train_student_ce(ResNet18Half(), 'ResNet-18-Half', epochs=200)
    resnet_half_kd = train_student_kd(
        teacher_resnet, ResNet18Half(), 'ResNet-18-Half', epochs=200
    )
    resnet_zskd_acc = resnet_zskd(teacher_resnet, epochs=300)

    print(f"\n{'='*60}")
    print(f"  TABLE 6 — COMPLETE (ResNet-18 Teacher)")
    print(f"{'='*60}")
    print(f"  {'Model':<30s} {'Ours':>8s}  {'Paper':>8s}")
    print(f"  {'-'*48}")
    print(f"  {'ResNet-18 Teacher':<30s} {f'{resnet_teacher_acc:.2f}%':>8s}  {'86.54%':>8s}")
    print(f"  {'ResNet-18-Half Student-CE':<30s} {f'{resnet_half_ce:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'ResNet-18-Half Student-KD':<30s} {f'{resnet_half_kd:.2f}%':>8s}  {'—':>8s}")
    print(f"  {'ResNet-18-Half ZSKD':<30s} {f'{resnet_zskd_acc:.2f}%':>8s}  {'81.10%':>8s}")
    print(f"{'='*60}")