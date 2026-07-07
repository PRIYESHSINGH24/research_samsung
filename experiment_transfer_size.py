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


def generate_dis_for_size(model, sim_matrix, num_per_class):
    print(f"\n  Generating {num_per_class * 10} DIs...")
    model.eval()
    all_di_images = []
    all_di_labels = []

    n_per_beta = num_per_class // len(config.BETA_VALUES)

    for class_idx in range(config.NUM_CLASSES):
        alpha_k = sim_matrix[class_idx]
        class_dis = []

        for beta in config.BETA_VALUES:
            sampled_vectors = sample_dirichlet_vectors(
                alpha_k, beta, n_per_beta
            )
            for start_idx in range(0, n_per_beta, config.DI_BATCH_SIZE):
                end_idx = min(start_idx + config.DI_BATCH_SIZE, n_per_beta)
                batch_targets = sampled_vectors[start_idx:end_idx]
                di_batch = generate_di_batch(
                    model, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_di_tensor = torch.cat(class_dis, dim=0)
        all_di_images.append(class_di_tensor)
        all_di_labels.extend([class_idx] * num_per_class)

    all_di_images = torch.cat(all_di_images, dim=0)
    all_di_labels = torch.tensor(all_di_labels, dtype=torch.long)
    print(f"  Generated: {all_di_images.shape}")
    return all_di_images, all_di_labels


def train_student_with_dis(teacher, di_images, di_labels, test_loader, epochs=300):
    di_dataset = AugmentedDIDataset(di_images, di_labels)
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
        for di_batch, _ in di_loader:
            di_batch = di_batch.to(config.DEVICE)

            with torch.no_grad():
                t_logits = teacher(di_batch)
            s_logits = student(di_batch)
            loss = kd_loss(s_logits, t_logits, config.KD_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate(student, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc

    return best_acc


def run_transfer_size_experiment():
    print("=" * 50)
    print("  EXPERIMENT: TRANSFER SET SIZE EFFECT")
    print("  Paper Section 4.1.4")
    print("=" * 50)

    # Teacher load
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

    # Paper percentages
    percentages = [1, 5, 10, 20, 40]
    num_per_class = [60, 300, 600, 1200, 2400]

    results = []

    for pct, npc in zip(percentages, num_per_class):
        print(f"\n{'=' * 40}")
        print(f"  Transfer Set: {pct}% ({npc * 10} DIs)")
        print(f"{'=' * 40}")

        # Generate DIs
        dis, labels = generate_dis_for_size(
            teacher, sim_matrix, npc
        )

        # Train student
        acc = train_student_with_dis(
            teacher, dis, labels, test_loader, epochs=300
        )

        results.append(acc)
        print(f"  ✅ {pct}%: Accuracy = {acc:.2f}%")

    # Results
    print(f"\n{'=' * 50}")
    print(f"  TRANSFER SET SIZE — RESULTS")
    print(f"{'=' * 50}")
    for pct, acc in zip(percentages, results):
        print(f"  {pct:3d}% ({pct * 600:5d} DIs) → {acc:.2f}%")
    print(f"{'=' * 50}")

    # Plot — Paper Figure 3
    plt.figure(figsize=(8, 5))
    plt.plot(percentages, results, 'bo-',
             linewidth=2, markersize=8,
             label='Data Impressions')
    plt.xlabel('Transfer Set Size (% of Training Data)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Effect of Transfer Set Size — MNIST')
    plt.xticks(percentages)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = f'{config.RESULTS_DIR}transfer_size_effect.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Plot saved: {path}")

    return percentages, results


if __name__ == "__main__":
    percentages, results = run_transfer_size_experiment()