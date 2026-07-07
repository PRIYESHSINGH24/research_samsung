import torch
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


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST(
        config.DATA_DIR, train=False,
        download=False, transform=transform
    )
    return DataLoader(
        test_data, batch_size=256,
        shuffle=False, num_workers=0,
        pin_memory=True
    )


def kd_loss(student_logits, teacher_logits, temperature):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(
        log_soft_student, soft_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss


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


def generate_class_impression_batch(model, class_idx, batch_size, device):
    """
    Class Impressions — One-hot target softmax
    Special case of DI where β → 0
    Target = [0, 0, ..., 1, ..., 0]
    """
    # One-hot target
    target = torch.zeros(batch_size, config.NUM_CLASSES, device=device)
    target[:, class_idx] = 1.0

    di_batch = torch.randn(
        batch_size, config.IN_CHANNELS,
        config.IMAGE_SIZE, config.IMAGE_SIZE,
        device=device
    )
    di_batch.requires_grad_(True)

    optimizer = torch.optim.Adam([di_batch], lr=config.DI_LR)

    for iteration in range(config.DI_ITERATIONS):
        optimizer.zero_grad()
        logits = model(di_batch, temperature=config.TEMPERATURE)
        pred_softmax = F.softmax(logits, dim=1)
        loss = -torch.mean(
            torch.sum(target * torch.log(pred_softmax + 1e-8), dim=1)
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            di_batch.clamp_(-2.5, 2.5)

    return di_batch.detach()


def generate_class_impressions(model, num_per_class):
    """Generate Class Impressions (one-hot targets)"""
    print(f"\n  Generating {num_per_class * 10} Class Impressions...")
    model.eval()
    all_images = []
    all_labels = []

    for class_idx in range(config.NUM_CLASSES):
        class_cis = []
        pbar = tqdm(
            range(0, num_per_class, config.DI_BATCH_SIZE),
            desc=f"  CI Class {class_idx}"
        )
        for start_idx in pbar:
            end_idx = min(start_idx + config.DI_BATCH_SIZE, num_per_class)
            batch_size = end_idx - start_idx
            ci_batch = generate_class_impression_batch(
                model, class_idx, batch_size, config.DEVICE
            )
            class_cis.append(ci_batch.cpu())

        class_ci_tensor = torch.cat(class_cis, dim=0)
        all_images.append(class_ci_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ CI Class {class_idx}: {class_ci_tensor.shape[0]} done")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Total CIs: {all_images.shape}")
    return all_images, all_labels


def generate_data_impressions_for_size(model, sim_matrix, num_per_class):
    """Generate Data Impressions (Dirichlet targets)"""
    print(f"\n  Generating {num_per_class * 10} Data Impressions...")
    model.eval()
    all_images = []
    all_labels = []
    n_per_beta = num_per_class // len(config.BETA_VALUES)

    for class_idx in range(config.NUM_CLASSES):
        alpha_k = sim_matrix[class_idx]
        class_dis = []

        for beta in config.BETA_VALUES:
            sampled_vectors = sample_dirichlet_vectors(
                alpha_k, beta, n_per_beta
            )
            pbar = tqdm(
                range(0, n_per_beta, config.DI_BATCH_SIZE),
                desc=f"  DI Class {class_idx} β={beta}"
            )
            for start_idx in pbar:
                end_idx = min(start_idx + config.DI_BATCH_SIZE, n_per_beta)
                batch_targets = sampled_vectors[start_idx:end_idx]
                di_batch = generate_di_batch(
                    model, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_di_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_di_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ DI Class {class_idx}: {class_di_tensor.shape[0]} done")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Total DIs: {all_images.shape}")
    return all_images, all_labels


def train_student(teacher, images, labels, test_loader, epochs=300):
    di_dataset = AugmentedDIDataset(images, labels)
    di_loader = DataLoader(
        di_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )

    student = LeNet5Half().to(config.DEVICE)
    optimizer = optim.Adam(
        student.parameters(), lr=config.STUDENT_LR,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    teacher.eval()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for batch, _ in di_loader:
            batch = batch.to(config.DEVICE)
            with torch.no_grad():
                t_logits = teacher(batch)
            s_logits = student(batch)
            loss = kd_loss(s_logits, t_logits, config.KD_TEMPERATURE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        test_acc = evaluate(student, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc


def run_class_vs_data_experiment():
    print("=" * 50)
    print("  EXPERIMENT: CLASS vs DATA IMPRESSIONS")
    print("  Paper Section 4.1.5")
    print("=" * 50)

    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    sim_matrix = compute_similarity_matrix(teacher)
    test_loader = get_test_loader()

    # Paper ke percentages
    percentages = [1, 5, 10, 20, 40]
    num_per_class_list = [60, 300, 600, 1200, 2400]

    ci_results = []
    di_results = []

    for pct, npc in zip(percentages, num_per_class_list):
        print(f"\n{'=' * 45}")
        print(f"  SIZE: {pct}% ({npc * 10} samples)")
        print(f"{'=' * 45}")

        # Class Impressions
        print(f"\n  --- CLASS IMPRESSIONS ---")
        ci_images, ci_labels = generate_class_impressions(teacher, npc)
        ci_acc = train_student(
            teacher, ci_images, ci_labels, test_loader
        )
        ci_results.append(ci_acc)
        print(f"  CI {pct}%: {ci_acc:.2f}%")

        # Data Impressions
        print(f"\n  --- DATA IMPRESSIONS ---")
        di_images, di_labels = generate_data_impressions_for_size(
            teacher, sim_matrix, npc
        )
        di_acc = train_student(
            teacher, di_images, di_labels, test_loader
        )
        di_results.append(di_acc)
        print(f"  DI {pct}%: {di_acc:.2f}%")

    # Results
    print(f"\n{'=' * 50}")
    print(f"  CLASS vs DATA IMPRESSIONS — RESULTS")
    print(f"{'=' * 50}")
    print(f"  {'Size':>5s}  {'CI':>8s}  {'DI':>8s}  {'Winner':>10s}")
    print(f"  {'-'*35}")
    for pct, ci, di in zip(percentages, ci_results, di_results):
        winner = "DI ✅" if di > ci else "CI"
        print(f"  {pct:4d}%  {ci:7.2f}%  {di:7.2f}%  {winner:>10s}")
    print(f"{'=' * 50}")

    # Plot — Paper Figure 4
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentages, ci_results, 'rs-',
            linewidth=2, markersize=8,
            label='Class Impressions')
    ax.plot(percentages, di_results, 'bo-',
            linewidth=2, markersize=8,
            label='Data Impressions (Ours)')
    ax.set_xlabel('Transfer Set Size (% of Training Data)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Class vs Data Impressions — MNIST')
    ax.set_xticks(percentages)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    path = f'{config.RESULTS_DIR}class_vs_data_impressions.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Plot saved: {path}")

    return ci_results, di_results


if __name__ == "__main__":
    ci_results, di_results = run_class_vs_data_experiment()