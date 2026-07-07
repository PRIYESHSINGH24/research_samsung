import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from config import config


# ═══════════════════════════════════
# RESNET-32 FOR CIFAR-100
# ═══════════════════════════════════

class BasicBlock32(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet32(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 5, stride=1)
        self.layer2 = self._make_layer(32, 5, stride=2)
        self.layer3 = self._make_layer(64, 5, stride=2)

        self.final_layer = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock32(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, temperature=1.0):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        logits = self.final_layer(out)
        return logits / temperature

    def get_final_weights(self):
        return self.final_layer.weight.data.clone()


# ═══════════════════════════════════
# DATA
# ═══════════════════════════════════

def get_cifar100_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])
    train_data = datasets.CIFAR100(
        'data/', train=True,
        download=True, transform=transform_train
    )
    test_data = datasets.CIFAR100(
        'data/', train=False,
        download=True, transform=transform_test
    )
    return train_data, test_data


def get_class_subset(dataset, classes):
    """Get subset of dataset containing only specified classes"""
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)


def get_class_loader(dataset, classes, batch_size=128, shuffle=True):
    subset = get_class_subset(dataset, classes)
    return DataLoader(
        subset, batch_size=batch_size,
        shuffle=shuffle, num_workers=0, pin_memory=True
    )


# ═══════════════════════════════════
# DI GENERATION FOR CONTINUAL LEARNING
# ═══════════════════════════════════

def generate_dis_for_classes(model, classes, num_per_class=500, temperature=20):
    """Generate DIs for specific classes using model's final layer weights"""
    model.eval()

    weights = model.get_final_weights()
    num_total_classes = weights.shape[0]

    all_images = []
    all_labels = []

    for class_idx in classes:
        # Compute similarity for this class
        w = weights[class_idx]
        cos_sim = F.cosine_similarity(
            w.unsqueeze(0), weights, dim=1
        )
        # Min-max normalize
        cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        alpha_k = cos_sim.cpu().numpy()

        class_dis = []

        for beta in [1.0, 0.1]:
            n_samples = num_per_class // 2

            # Dirichlet sampling
            alpha = alpha_k * beta
            alpha = np.clip(alpha, 1e-3, None)
            targets = np.random.dirichlet(alpha, n_samples)
            targets = torch.tensor(targets, dtype=torch.float32)

            for start in range(0, n_samples, 32):
                end = min(start + 32, n_samples)
                batch_targets = targets[start:end].to(config.DEVICE)
                B = batch_targets.shape[0]

                di_batch = torch.randn(
                    B, 3, 32, 32,
                    device=config.DEVICE
                )
                di_batch.requires_grad_(True)
                optimizer = torch.optim.Adam([di_batch], lr=0.1)

                for it in range(1000):
                    optimizer.zero_grad()
                    logits = model(di_batch, temperature=temperature)
                    pred_softmax = F.softmax(logits, dim=1)
                    loss = -torch.mean(
                        torch.sum(batch_targets * torch.log(pred_softmax + 1e-8), dim=1)
                    )
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        di_batch.clamp_(-2.5, 2.5)

                class_dis.append(di_batch.detach().cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    return all_images, all_labels


# ═══════════════════════════════════
# EVALUATION
# ═══════════════════════════════════

def evaluate_classes(model, test_data, classes):
    """Evaluate model on specific classes"""
    loader = get_class_loader(test_data, classes, batch_size=128, shuffle=False)
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
    if total == 0:
        return 0.0
    return 100. * correct / total


def evaluate_all_seen(model, test_data, all_seen_classes):
    """Evaluate on all classes seen so far"""
    return evaluate_classes(model, test_data, all_seen_classes)


# ═══════════════════════════════════
# CONTINUAL LEARNING WITH DIs
# ═══════════════════════════════════

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def continual_learning_experiment():
    print("="*60)
    print("  SECTION 4.3 — CONTINUAL LEARNING")
    print("  Dataset: CIFAR-100")
    print("  Model: ResNet-32")
    print("  Incremental Steps: 20 classes each")
    print("="*60)

    train_data, test_data = get_cifar100_loaders()
    print(f"\n  CIFAR-100 Train: {len(train_data):,}")
    print(f"  CIFAR-100 Test : {len(test_data):,}")

    # Class order (fixed for reproducibility)
    all_classes = list(range(100))
    np.random.seed(42)
    np.random.shuffle(all_classes)

    step_size = 20
    num_steps = 100 // step_size  # 5 steps

    # Results storage
    step_results = []  # Accuracy after each step on ALL seen classes
    new_class_results = []  # Accuracy on just the NEW classes
    old_class_results = []  # Accuracy on OLD classes (forgetting check)

    model = ResNet32(num_classes=100).to(config.DEVICE)
    all_seen_classes = []
    stored_dis = None
    stored_labels = None

    for step in range(num_steps):
        start_class = step * step_size
        end_class = (step + 1) * step_size
        new_classes = all_classes[start_class:end_class]
        old_classes = list(all_seen_classes)  # Copy before updating
        all_seen_classes.extend(new_classes)

        print(f"\n{'='*50}")
        print(f"  STEP {step+1}/{num_steps}")
        print(f"  New Classes: {new_classes[:5]}...{new_classes[-5:]}")
        print(f"  Total Seen : {len(all_seen_classes)}")
        print(f"{'='*50}")

        # ── Build training data ──
        # New classes: real data
        new_loader = get_class_loader(train_data, new_classes, batch_size=128)

        # Old classes: DIs (if any)
        if stored_dis is not None:
            print(f"  Using {len(stored_dis)} DIs for old classes")
            old_dataset = AugmentedDataset(stored_dis, stored_labels)
            old_loader = DataLoader(
                old_dataset, batch_size=128,
                shuffle=True, num_workers=0, pin_memory=True
            )
        else:
            old_loader = None

        # ── Train on new + old(DI) ──
        optimizer = optim.SGD(
            model.parameters(), lr=0.1,
            momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=70, eta_min=1e-4
        )
        criterion = nn.CrossEntropyLoss()

        epochs = 70

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            n_batches = 0

            # Train on new classes (real data)
            for images, labels in new_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Train on old classes (DIs)
            if old_loader is not None:
                for di_batch, di_labels in old_loader:
                    di_batch = di_batch.to(config.DEVICE)
                    di_labels = di_labels.to(config.DEVICE)

                    optimizer.zero_grad()
                    outputs = model(di_batch)
                    loss = criterion(outputs, di_labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == 1:
                all_acc = evaluate_all_seen(model, test_data, all_seen_classes)
                print(f"  Epoch [{epoch:2d}/{epochs}] "
                      f"Loss: {total_loss/max(n_batches,1):.4f} | "
                      f"All Seen Acc: {all_acc:.2f}%")

        # ── Evaluate ──
        all_acc = evaluate_all_seen(model, test_data, all_seen_classes)
        new_acc = evaluate_classes(model, test_data, new_classes)

        step_results.append(all_acc)
        new_class_results.append(new_acc)

        if old_classes:
            old_acc = evaluate_classes(model, test_data, old_classes)
            old_class_results.append(old_acc)
            print(f"\n  Results Step {step+1}:")
            print(f"    New Classes Acc : {new_acc:.2f}%")
            print(f"    Old Classes Acc : {old_acc:.2f}%")
            print(f"    All Seen Acc    : {all_acc:.2f}%")
        else:
            old_class_results.append(0)
            print(f"\n  Results Step {step+1}:")
            print(f"    New Classes Acc : {new_acc:.2f}%")
            print(f"    All Seen Acc    : {all_acc:.2f}%")

        # ── Generate DIs for ALL seen classes ──
        print(f"\n  Generating DIs for {len(all_seen_classes)} seen classes...")
        stored_dis, stored_labels = generate_dis_for_classes(
            model, all_seen_classes,
            num_per_class=200,  # 200 per class to save time
            temperature=20
        )
        print(f"  ✅ Generated {len(stored_dis)} DIs")

        # Save checkpoint
        torch.save({
            'model': model.state_dict(),
            'step': step + 1,
            'all_seen_classes': all_seen_classes,
            'step_results': step_results,
        }, f'{config.CHECKPOINT_DIR}continual_step{step+1}.pth')

    # ═══════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TABLE 8 — CONTINUAL LEARNING RESULTS")
    print(f"{'='*60}")
    print(f"  {'Step':<6s} {'Classes':<12s} {'New Acc':>10s} {'Old Acc':>10s} {'All Acc':>10s}")
    print(f"  {'-'*50}")

    for step in range(num_steps):
        n_classes = (step + 1) * step_size
        new_a = new_class_results[step]
        old_a = old_class_results[step]
        all_a = step_results[step]
        print(f"  {step+1:<6d} {n_classes:<12d} {new_a:>9.2f}% {old_a:>9.2f}% {all_a:>9.2f}%")

    print(f"{'='*60}")

    # Average Incremental Accuracy
    avg_inc = np.mean(step_results)
    print(f"\n  Average Incremental Accuracy: {avg_inc:.2f}%")
    print(f"  Paper Target                : ~55-60%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps_x = [(i+1)*step_size for i in range(num_steps)]

    # All seen accuracy
    ax1.plot(steps_x, step_results, 'bo-', linewidth=2, markersize=8, label='ZSKD (Ours)')
    ax1.set_xlabel('Number of Classes Seen')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Incremental Accuracy on All Seen Classes')
    ax1.legend()
    ax1.grid(True)

    # New vs Old
    ax2.plot(steps_x, new_class_results, 'go-', linewidth=2, markersize=8, label='New Classes')
    ax2.plot(steps_x[1:], old_class_results[1:], 'rs-', linewidth=2, markersize=8, label='Old Classes')
    ax2.set_xlabel('Number of Classes Seen')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('New vs Old Class Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Continual Learning — CIFAR-100 with Data Impressions')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}continual_learning.png', dpi=150)
    plt.close()
    print(f"  ✅ Plot saved!")

    return step_results


if __name__ == "__main__":
    results = continual_learning_experiment()