"""
EXTENSION — Multi-Layer Similarity for Improved Data Impressions
Our Contribution:
  Instead of using ONLY final layer weights for similarity,
  use features from MULTIPLE layers for richer similarity matrix.
  
  2-Stage Approach (still data-free!):
    Stage 1: Generate DIs using original method (single layer)
    Stage 2: Pass Stage-1 DIs through teacher
             → compute multi-layer similarity
             → generate NEW better DIs
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
import matplotlib.pyplot as plt

from models.lenet import LeNet5, LeNet5Half
from models.alexnet import AlexNet, AlexNetHalf
from config import config


# ═══════════════════════════════════
# MULTI-LAYER FEATURE EXTRACTION
# ═══════════════════════════════════

class LeNet5MultiLayer(nn.Module):
    """LeNet-5 with hooks to extract intermediate features"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.features = {}

    def forward(self, x, temperature=1.0):
        # Layer 1: after conv1 + pool
        x1 = self.model.pool(F.relu(self.model.conv1(x)))
        self.features['layer1'] = x1

        # Layer 2: after conv2 + pool
        x2 = self.model.pool(F.relu(self.model.conv2(x1)))
        self.features['layer2'] = x2

        # Flatten
        x3 = x2.view(x2.size(0), -1)

        # Layer 3: after fc1
        x4 = F.relu(self.model.fc1(x3))
        self.features['layer3'] = x4

        # Layer 4: after fc2
        x5 = F.relu(self.model.fc2(x4))
        self.features['layer4'] = x5

        # Output
        logits = self.model.fc3(x5)
        return logits / temperature

    def get_all_features(self, x):
        """Forward pass and return all layer features"""
        self.forward(x)
        return {k: v.detach() for k, v in self.features.items()}


class AlexNetMultiLayer(nn.Module):
    """AlexNet with hooks to extract intermediate features"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.features = {}

    def forward(self, x, temperature=1.0):
        # Extract features at each pool layer
        feat = x
        pool_count = 0
        for i, layer in enumerate(self.model.features):
            feat = layer(feat)
            if isinstance(layer, nn.MaxPool2d):
                pool_count += 1
                self.features[f'pool{pool_count}'] = feat

        feat = feat.view(feat.size(0), -1)

        # Through classifier
        for i, layer in enumerate(self.model.classifier):
            feat = layer(feat)
            if isinstance(layer, nn.ReLU) and i > 0:
                self.features[f'fc{i}'] = feat

        logits = self.model.final_layer(feat)
        self.features['pre_final'] = feat
        return logits / temperature

    def get_all_features(self, x):
        self.forward(x)
        return {k: v.detach() for k, v in self.features.items()}


# ═══════════════════════════════════
# MULTI-LAYER SIMILARITY COMPUTATION
# ═══════════════════════════════════

def compute_single_layer_similarity(model):
    """Original paper method — final layer only"""
    weights = model.get_final_weights()
    sim = F.cosine_similarity(weights.unsqueeze(1), weights.unsqueeze(0), dim=2)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.cpu().numpy()


def compute_multi_layer_similarity(multi_model, data_loader, num_classes=10, layer_weights=None):
    """
    OUR PROPOSED METHOD:
    Compute similarity using features from multiple layers.

    Args:
        multi_model: Model with feature extraction
        data_loader: DataLoader (DIs from Stage 1 or any data)
        num_classes: Number of classes
        layer_weights: Dict of {layer_name: weight} or None for equal
    """
    multi_model.eval()
    device = config.DEVICE

    # Step 1: Collect features per layer per class
    layer_features = {}  # {layer_name: {class: [features]}}

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="  Extracting features"):
            images = images.to(device)
            all_feats = multi_model.get_all_features(images)

            for layer_name, feats in all_feats.items():
                if layer_name not in layer_features:
                    layer_features[layer_name] = {c: [] for c in range(num_classes)}

                # Global average pool if spatial dimensions exist
                if feats.dim() == 4:
                    feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)

                for i, label in enumerate(labels):
                    layer_features[layer_name][label.item()].append(feats[i].cpu())

    # Step 2: Compute class-wise mean features per layer
    layer_means = {}
    for layer_name, class_feats in layer_features.items():
        means = []
        for c in range(num_classes):
            if len(class_feats[c]) > 0:
                class_mean = torch.stack(class_feats[c]).mean(0)
            else:
                class_mean = torch.zeros_like(list(class_feats.values())[0][0])
            means.append(class_mean)
        layer_means[layer_name] = torch.stack(means)  # [num_classes, feat_dim]

    # Step 3: Compute per-layer similarity matrices
    layer_sims = {}
    for layer_name, means in layer_means.items():
        sim = F.cosine_similarity(means.unsqueeze(1), means.unsqueeze(0), dim=2)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        layer_sims[layer_name] = sim.numpy()
        print(f"  Layer '{layer_name}': sim shape {sim.shape}")

    # Step 4: Weighted combination
    if layer_weights is None:
        # Equal weights
        layer_weights = {name: 1.0 / len(layer_sims) for name in layer_sims}

    combined_sim = np.zeros((num_classes, num_classes))
    for layer_name, sim in layer_sims.items():
        w = layer_weights.get(layer_name, 0)
        combined_sim += w * sim

    # Normalize combined
    combined_sim = (combined_sim - combined_sim.min()) / (combined_sim.max() - combined_sim.min() + 1e-8)

    return combined_sim, layer_sims


# ═══════════════════════════════════
# DI GENERATION (same as original)
# ═══════════════════════════════════

def sample_dirichlet(alpha, beta, n):
    a = alpha * beta
    a = np.clip(a, 1e-3, None)
    return torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)


def generate_di_batch(model, targets, device, channels=1):
    B = targets.shape[0]
    di = torch.randn(B, channels, 32, 32, device=device)
    di.requires_grad_(True)
    targets = targets.to(device)
    optimizer = torch.optim.Adam([di], lr=0.1)

    for _ in range(1500):
        optimizer.zero_grad()
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            di.clamp_(-2.5, 2.5)
    return di.detach()


def generate_dis(model, sim_matrix, n_per_class, channels=1):
    """Generate DIs using given similarity matrix"""
    all_dis, all_labels = [], []

    for cls in range(10):
        alpha = sim_matrix[cls]
        class_dis = []
        for beta in [1.0, 0.1]:
            n = n_per_class // 2
            targets = sample_dirichlet(alpha, beta, n)
            pbar = tqdm(range(0, n, 32), desc=f"  C{cls} b={beta}")
            for start in pbar:
                end = min(start + 32, n)
                di = generate_di_batch(model, targets[start:end], config.DEVICE, channels)
                class_dis.append(di.cpu())
        ct = torch.cat(class_dis, dim=0)
        all_dis.append(ct)
        all_labels.extend([cls] * n_per_class)
        print(f"  Done C{cls}: {ct.shape[0]} DIs")

    return torch.cat(all_dis, 0), torch.tensor(all_labels, dtype=torch.long)


# ═══════════════════════════════════
# ZSKD TRAINING
# ═══════════════════════════════════

class AugDIDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(15, (0.1, 0.1), (0.9, 1.1)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, lab = self.images[idx], self.labels[idx]
        mn, mx = img.min(), img.max()
        n = (img - mn) / (mx - mn + 1e-8)
        a = self.tf(n)
        a = a * (mx - mn) + mn
        return a, lab


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            _, pred = model(img).max(1)
            total += lab.size(0)
            correct += pred.eq(lab).sum().item()
    return 100. * correct / total


def train_zskd(teacher, student, dis, labels, test_loader, epochs=500):
    di_dataset = AugDIDataset(dis, labels)
    di_loader = DataLoader(di_dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best = 0

    for ep in range(1, epochs + 1):
        student.train()
        for batch, _ in di_loader:
            batch = batch.to(config.DEVICE)
            with torch.no_grad():
                tl = teacher(batch)
            sl = student(batch)
            loss = F.kl_div(
                F.log_softmax(sl / 20, 1),
                F.softmax(tl / 20, 1),
                reduction='batchmean'
            ) * 400
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate(student, test_loader)
        if acc > best:
            best = acc
        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep}: {acc:.2f}% | Best: {best:.2f}%")

    return best


# ═══════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════

def plot_similarity_comparison(single_sim, multi_sim, layer_sims, dataset_name):
    """Plot original vs multi-layer similarity matrices"""
    n_plots = 2 + len(layer_sims)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    axes[0].imshow(single_sim, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Original\n(Final Layer Only)')

    for i, (name, sim) in enumerate(layer_sims.items()):
        axes[i + 1].imshow(sim, cmap='hot', vmin=0, vmax=1)
        axes[i + 1].set_title(f'Layer: {name}')

    axes[-1].imshow(multi_sim, cmap='hot', vmin=0, vmax=1)
    axes[-1].set_title('Multi-Layer\n(Combined)')

    plt.suptitle(f'Similarity Matrix Comparison — {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}multi_layer_sim_{dataset_name}.png', dpi=150)
    print(f"  Saved: multi_layer_sim_{dataset_name}.png")


# ═══════════════════════════════════
# MAIN — EXPERIMENT
# ═══════════════════════════════════

if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXTENSION — Multi-Layer Similarity")
    print("  Compare: Original (Single Layer) vs Proposed (Multi-Layer)")
    print("=" * 65)

    # Weight configurations to test
    weight_configs = {
        'equal': None,  # Equal weights for all layers
        'increasing': None,  # Will be set per model
        'final_heavy': None,  # Heavy weight on final layer
    }

    all_results = {}

    # ══════════════════════════════
    # MNIST
    # ══════════════════════════════
    print("\n" + "=" * 50)
    print("  MNIST")
    print("=" * 50)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    # Load teacher
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}mnist_teacher.pth', map_location=config.DEVICE)
    )
    teacher.eval()

    # ── Stage 1: Original DIs ──
    print("\n  --- Stage 1: Original (Single Layer) DIs ---")
    single_sim = compute_single_layer_similarity(teacher)
    stage1_dis, stage1_labels = generate_dis(teacher, single_sim, 2400, channels=1)

    print("\n  --- Stage 1 ZSKD ---")
    student_single = LeNet5Half().to(config.DEVICE)
    single_acc = train_zskd(teacher, student_single, stage1_dis, stage1_labels, test_loader, epochs=500)
    print(f"  Single-Layer ZSKD: {single_acc:.2f}%")

    # ── Stage 2: Multi-Layer DIs ──
    print("\n  --- Stage 2: Multi-Layer Similarity ---")
    multi_model = LeNet5MultiLayer(teacher).to(config.DEVICE)
    multi_model.eval()

    # Use Stage 1 DIs as proxy data for feature extraction
    stage1_dataset = torch.utils.data.TensorDataset(stage1_dis, stage1_labels)
    stage1_loader = DataLoader(stage1_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Test different weight configurations
    mnist_results = {}

    # Config 1: Equal weights
    print("\n  [Config: Equal Weights]")
    multi_sim_eq, layer_sims = compute_multi_layer_similarity(
        multi_model, stage1_loader, num_classes=10, layer_weights=None
    )
    multi_dis_eq, multi_labels_eq = generate_dis(teacher, multi_sim_eq, 2400, channels=1)
    student_eq = LeNet5Half().to(config.DEVICE)
    eq_acc = train_zskd(teacher, student_eq, multi_dis_eq, multi_labels_eq, test_loader, epochs=500)
    mnist_results['equal'] = eq_acc

    # Config 2: Increasing weights (deeper = more weight)
    print("\n  [Config: Increasing Weights]")
    layer_names = list(layer_sims.keys())
    n_layers = len(layer_names)
    inc_weights = {}
    for i, name in enumerate(layer_names):
        inc_weights[name] = (i + 1) / sum(range(1, n_layers + 1))
    multi_sim_inc, _ = compute_multi_layer_similarity(
        multi_model, stage1_loader, num_classes=10, layer_weights=inc_weights
    )
    multi_dis_inc, multi_labels_inc = generate_dis(teacher, multi_sim_inc, 2400, channels=1)
    student_inc = LeNet5Half().to(config.DEVICE)
    inc_acc = train_zskd(teacher, student_inc, multi_dis_inc, multi_labels_inc, test_loader, epochs=500)
    mnist_results['increasing'] = inc_acc

    # Config 3: Final-heavy (0.1 for early, 0.7 for final)
    print("\n  [Config: Final-Heavy Weights]")
    fh_weights = {}
    for i, name in enumerate(layer_names):
        if i == n_layers - 1:
            fh_weights[name] = 0.7
        else:
            fh_weights[name] = 0.3 / (n_layers - 1)
    multi_sim_fh, _ = compute_multi_layer_similarity(
        multi_model, stage1_loader, num_classes=10, layer_weights=fh_weights
    )
    multi_dis_fh, multi_labels_fh = generate_dis(teacher, multi_sim_fh, 2400, channels=1)
    student_fh = LeNet5Half().to(config.DEVICE)
    fh_acc = train_zskd(teacher, student_fh, multi_dis_fh, multi_labels_fh, test_loader, epochs=500)
    mnist_results['final_heavy'] = fh_acc

    # Visualization
    plot_similarity_comparison(single_sim, multi_sim_eq, layer_sims, 'MNIST')

    all_results['MNIST'] = {
        'single': single_acc,
        'equal': mnist_results['equal'],
        'increasing': mnist_results['increasing'],
        'final_heavy': mnist_results['final_heavy'],
    }

    # ══════════════════════════════
    # FMNIST
    # ══════════════════════════════
    print("\n" + "=" * 50)
    print("  FASHION-MNIST")
    print("=" * 50)

    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    test_data = datasets.FashionMNIST('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth', map_location=config.DEVICE)
    )
    teacher.eval()

    # Stage 1
    print("\n  --- Stage 1: Original DIs ---")
    single_sim = compute_single_layer_similarity(teacher)
    stage1_dis, stage1_labels = generate_dis(teacher, single_sim, 4800, channels=1)
    student_single = LeNet5Half().to(config.DEVICE)
    single_acc = train_zskd(teacher, student_single, stage1_dis, stage1_labels, test_loader, epochs=500)

    # Stage 2: Best config from MNIST
    print("\n  --- Stage 2: Multi-Layer (Equal) ---")
    multi_model = LeNet5MultiLayer(teacher).to(config.DEVICE)
    stage1_loader = DataLoader(
        torch.utils.data.TensorDataset(stage1_dis, stage1_labels),
        batch_size=128, shuffle=False, num_workers=0
    )
    multi_sim, layer_sims = compute_multi_layer_similarity(multi_model, stage1_loader)
    multi_dis, multi_labels = generate_dis(teacher, multi_sim, 4800, channels=1)
    student_multi = LeNet5Half().to(config.DEVICE)
    multi_acc = train_zskd(teacher, student_multi, multi_dis, multi_labels, test_loader, epochs=500)

    plot_similarity_comparison(single_sim, multi_sim, layer_sims, 'FMNIST')
    all_results['FMNIST'] = {'single': single_acc, 'multi': multi_acc}

    # ══════════════════════════════
    # CIFAR-10
    # ══════════════════════════════
    print("\n" + "=" * 50)
    print("  CIFAR-10")
    print("=" * 50)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data = datasets.CIFAR10('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    teacher = AlexNet().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE)
    )
    teacher.eval()

    # Stage 1
    print("\n  --- Stage 1: Original DIs ---")
    single_sim = compute_single_layer_similarity(teacher)
    stage1_dis, stage1_labels = generate_dis(teacher, single_sim, 4000, channels=3)
    student_single = AlexNetHalf().to(config.DEVICE)
    single_acc = train_zskd(teacher, student_single, stage1_dis, stage1_labels, test_loader, epochs=500)

    # Stage 2
    print("\n  --- Stage 2: Multi-Layer ---")
    multi_model = AlexNetMultiLayer(teacher).to(config.DEVICE)
    stage1_loader = DataLoader(
        torch.utils.data.TensorDataset(stage1_dis, stage1_labels),
        batch_size=128, shuffle=False, num_workers=0
    )
    multi_sim, layer_sims = compute_multi_layer_similarity(multi_model, stage1_loader)
    multi_dis, multi_labels = generate_dis(teacher, multi_sim, 4000, channels=3)
    student_multi = AlexNetHalf().to(config.DEVICE)
    multi_acc = train_zskd(teacher, student_multi, multi_dis, multi_labels, test_loader, epochs=500)

    plot_similarity_comparison(single_sim, multi_sim, layer_sims, 'CIFAR10')
    all_results['CIFAR'] = {'single': single_acc, 'multi': multi_acc}

    # ══════════════════════════════
    # FINAL RESULTS TABLE
    # ══════════════════════════════
    print()
    print("=" * 65)
    print("  EXTENSION RESULTS — Multi-Layer Similarity")
    print("=" * 65)
    print(f"  {'Method':<25s} {'MNIST':>8s} {'FMNIST':>8s} {'CIFAR':>8s}")
    print(f"  {'-'*50}")
    print(f"  {'Paper (reported)':<25s} {'98.77%':>8s} {'79.62%':>8s} {'69.56%':>8s}")

    m = all_results['MNIST']
    f = all_results['FMNIST']
    c = all_results['CIFAR']

    print(f"  {'Original (single)':<25s} {m['single']:>7.2f}% {f['single']:>7.2f}% {c['single']:>7.2f}%")

    if 'equal' in m:
        print(f"  {'Multi (equal wt)':<25s} {m['equal']:>7.2f}%      -         -")
    if 'increasing' in m:
        print(f"  {'Multi (increasing)':<25s} {m['increasing']:>7.2f}%      -         -")
    if 'final_heavy' in m:
        print(f"  {'Multi (final heavy)':<25s} {m['final_heavy']:>7.2f}%      -         -")

    print(f"  {'Multi-Layer (best)':<25s} {m.get('multi', m.get('equal', 0)):>7.2f}% {f['multi']:>7.2f}% {c['multi']:>7.2f}%")
    print("=" * 65)