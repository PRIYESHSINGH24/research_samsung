import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
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
# MODELS FOR DOMAIN ADAPTATION
# ═══════════════════════════════════

class FeatureExtractor(nn.Module):
    """LeNet-5 without final layer — extracts features"""
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Classifier(nn.Module):
    """Final classification layer"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x, temperature=1.0):
        return self.fc(x) / temperature


class Discriminator(nn.Module):
    """ADDA discriminator — distinguishes source vs target features"""
    def __init__(self, feature_dim=84):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════

def get_svhn_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.SVHN(
        'data/', split='train',
        download=False, transform=transform
    )
    test_data = datasets.SVHN(
        'data/', split='test',
        download=False, transform=transform
    )
    train_loader = DataLoader(
        train_data, batch_size=128,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=128,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"  SVHN Train: {len(train_data):,}")
    print(f"  SVHN Test : {len(test_data):,}")
    return train_loader, test_loader


def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
        train_data, batch_size=128,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=128,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"  MNIST Train: {len(train_data):,}")
    print(f"  MNIST Test : {len(test_data):,}")
    return train_loader, test_loader


def get_usps_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.USPS(
        'data/', train=True,
        download=False, transform=transform
    )
    test_data = datasets.USPS(
        'data/', train=False,
        download=False, transform=transform
    )
    train_loader = DataLoader(
        train_data, batch_size=128,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=128,
        shuffle=False, num_workers=0, pin_memory=True
    )
    print(f"  USPS Train: {len(train_data):,}")
    print(f"  USPS Test : {len(test_data):,}")
    return train_loader, test_loader


# ═══════════════════════════════════
# EVALUATION
# ═══════════════════════════════════

def evaluate_with_parts(feat_ext, classifier, loader):
    feat_ext.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            features = feat_ext(images)
            out = classifier(features)
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# ═══════════════════════════════════
# STEP 1: TRAIN SOURCE TEACHER
# ═══════════════════════════════════

def train_source_teacher(source_name, train_loader, test_loader, epochs=100):
    print(f"\n{'='*50}")
    print(f"  TRAINING {source_name} TEACHER")
    print(f"{'='*50}")

    feat_ext = FeatureExtractor(in_channels=1).to(config.DEVICE)
    classifier = Classifier(num_classes=10).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    params = list(feat_ext.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )

    best_acc = 0

    for epoch in range(1, epochs + 1):
        feat_ext.train()
        classifier.train()

        for images, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            features = feat_ext(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_acc = evaluate_with_parts(feat_ext, classifier, test_loader)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'feat_ext': feat_ext.state_dict(),
                'classifier': classifier.state_dict(),
            }, f'{config.CHECKPOINT_DIR}{source_name}_teacher_da.pth')

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  {source_name} Teacher Best: {best_acc:.2f}%")
    return feat_ext, classifier, best_acc


# ═══════════════════════════════════
# STEP 2: GENERATE DIs FROM SOURCE TEACHER
# ═══════════════════════════════════

def generate_dis_from_teacher(feat_ext, classifier, source_name, num_per_class=2400):
    print(f"\n{'='*50}")
    print(f"  GENERATING DIs FROM {source_name} TEACHER")
    print(f"{'='*50}")

    # Build full model for DI generation
    class FullModel(nn.Module):
        def __init__(self, feat_ext, classifier):
            super().__init__()
            self.feat_ext = feat_ext
            self.classifier = classifier

        def forward(self, x, temperature=1.0):
            features = self.feat_ext(x)
            return self.classifier(features, temperature)

        def get_final_weights(self):
            return self.classifier.fc.weight.data.clone()

    model = FullModel(feat_ext, classifier).to(config.DEVICE)
    model.eval()

    sim_matrix = compute_similarity_matrix(model)

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
                    model, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ Class {class_idx}: {class_tensor.shape[0]} DIs")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    save_path = f'{config.CHECKPOINT_DIR}{source_name}_dis_da.pth'
    torch.save({
        'di_images': all_images,
        'di_labels': all_labels,
    }, save_path)

    print(f"  Total: {all_images.shape}")
    print(f"  ✅ Saved: {save_path}")
    return all_images, all_labels


# ═══════════════════════════════════
# STEP 3: ADDA DOMAIN ADAPTATION
# ═══════════════════════════════════

def adda_adapt(source_feat_ext, source_classifier,
               di_images, target_loader, target_test_loader,
               source_name, target_name, epochs=100):
    """
    ADDA: Adversarial Discriminative Domain Adaptation
    Source: DIs (from source teacher)
    Target: Real target data (unlabeled)
    """
    print(f"\n{'='*50}")
    print(f"  ADDA: {source_name} → {target_name}")
    print(f"{'='*50}")

    # Target encoder (initialized from source)
    target_feat_ext = FeatureExtractor(in_channels=1).to(config.DEVICE)
    target_feat_ext.load_state_dict(source_feat_ext.state_dict())

    # Discriminator
    discriminator = Discriminator(feature_dim=84).to(config.DEVICE)

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    t_optimizer = optim.Adam(target_feat_ext.parameters(), lr=0.0002, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    # DI loader (source proxy)
    di_dataset = TensorDataset(di_images, torch.zeros(len(di_images)))
    di_loader = DataLoader(
        di_dataset, batch_size=128,
        shuffle=True, num_workers=0, pin_memory=True
    )

    source_feat_ext.eval()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        target_feat_ext.train()
        discriminator.train()

        di_iter = iter(di_loader)
        target_iter = iter(target_loader)

        n_batches = min(len(di_loader), len(target_loader))

        for _ in range(n_batches):
            # Get source (DI) batch
            try:
                di_batch, _ = next(di_iter)
            except StopIteration:
                di_iter = iter(di_loader)
                di_batch, _ = next(di_iter)

            # Get target batch
            try:
                target_batch, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch, _ = next(target_iter)

            di_batch = di_batch.to(config.DEVICE)
            target_batch = target_batch.to(config.DEVICE)
            B = min(di_batch.size(0), target_batch.size(0))
            di_batch = di_batch[:B]
            target_batch = target_batch[:B]

            # ── Train Discriminator ──
            with torch.no_grad():
                source_features = source_feat_ext(di_batch)
                target_features = target_feat_ext(target_batch)

            source_pred = discriminator(source_features)
            target_pred = discriminator(target_features)

            d_loss = criterion(source_pred, torch.ones(B, 1, device=config.DEVICE)) + \
                     criterion(target_pred, torch.zeros(B, 1, device=config.DEVICE))

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ── Train Target Encoder ──
            target_features = target_feat_ext(target_batch)
            target_pred = discriminator(target_features)

            # Fool discriminator — target should look like source
            t_loss = criterion(target_pred, torch.ones(B, 1, device=config.DEVICE))

            t_optimizer.zero_grad()
            t_loss.backward()
            t_optimizer.step()

        # Evaluate on target test set
        target_acc = evaluate_with_parts(
            target_feat_ext, source_classifier, target_test_loader
        )

        if target_acc > best_acc:
            best_acc = target_acc

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"D_loss: {d_loss.item():.4f} | "
                  f"Target Acc: {target_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")

    print(f"\n  {source_name}→{target_name} Best: {best_acc:.2f}%")
    return best_acc


# ═══════════════════════════════════
# MAIN — ALL 3 EXPERIMENTS
# ═══════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  SECTION 4.2 — DOMAIN ADAPTATION")
    print("  Using ADDA with Data Impressions")
    print("="*60)

    results = {}

    # ═══════════════════════════════════
    # EXPERIMENT 1: SVHN → MNIST
    # ═══════════════════════════════════
    print("\n" + "="*60)
    print("  EXPERIMENT 1: SVHN → MNIST")
    print("="*60)

    # Train SVHN teacher
    svhn_train_loader, svhn_test_loader = get_svhn_loaders()
    svhn_feat, svhn_clf, svhn_acc = train_source_teacher(
        'SVHN', svhn_train_loader, svhn_test_loader, epochs=100
    )

    # Load best
    ckpt = torch.load(f'{config.CHECKPOINT_DIR}SVHN_teacher_da.pth',
                       map_location=config.DEVICE)
    svhn_feat = FeatureExtractor(in_channels=1).to(config.DEVICE)
    svhn_feat.load_state_dict(ckpt['feat_ext'])
    svhn_clf = Classifier(num_classes=10).to(config.DEVICE)
    svhn_clf.load_state_dict(ckpt['classifier'])

    # Generate DIs from SVHN teacher
    svhn_dis, svhn_labels = generate_dis_from_teacher(
        svhn_feat, svhn_clf, 'SVHN', num_per_class=2400
    )

    # ADDA: SVHN → MNIST
    _, mnist_test_loader = get_mnist_loaders()
    mnist_train_loader, _ = get_mnist_loaders()

    svhn_to_mnist = adda_adapt(
        svhn_feat, svhn_clf,
        svhn_dis, mnist_train_loader, mnist_test_loader,
        'SVHN', 'MNIST', epochs=100
    )
    results['SVHN→MNIST'] = svhn_to_mnist

    # ═══════════════════════════════════
    # EXPERIMENT 2: MNIST → USPS
    # ═══════════════════════════════════
    print("\n" + "="*60)
    print("  EXPERIMENT 2: MNIST → USPS")
    print("="*60)

    # Train MNIST teacher
    mnist_train_loader, mnist_test_loader = get_mnist_loaders()
    mnist_feat, mnist_clf, mnist_acc = train_source_teacher(
        'MNIST', mnist_train_loader, mnist_test_loader, epochs=100
    )

    # Load best
    ckpt = torch.load(f'{config.CHECKPOINT_DIR}MNIST_teacher_da.pth',
                       map_location=config.DEVICE)
    mnist_feat = FeatureExtractor(in_channels=1).to(config.DEVICE)
    mnist_feat.load_state_dict(ckpt['feat_ext'])
    mnist_clf = Classifier(num_classes=10).to(config.DEVICE)
    mnist_clf.load_state_dict(ckpt['classifier'])

    # Generate DIs from MNIST teacher
    mnist_dis, mnist_labels = generate_dis_from_teacher(
        mnist_feat, mnist_clf, 'MNIST', num_per_class=2400
    )

    # ADDA: MNIST → USPS
    usps_train_loader, usps_test_loader = get_usps_loaders()

    mnist_to_usps = adda_adapt(
        mnist_feat, mnist_clf,
        mnist_dis, usps_train_loader, usps_test_loader,
        'MNIST', 'USPS', epochs=100
    )
    results['MNIST→USPS'] = mnist_to_usps

    # ═══════════════════════════════════
    # EXPERIMENT 3: USPS → MNIST
    # ═══════════════════════════════════
    print("\n" + "="*60)
    print("  EXPERIMENT 3: USPS → MNIST")
    print("="*60)

    # Train USPS teacher
    usps_train_loader, usps_test_loader = get_usps_loaders()
    usps_feat, usps_clf, usps_acc = train_source_teacher(
        'USPS', usps_train_loader, usps_test_loader, epochs=100
    )

    # Load best
    ckpt = torch.load(f'{config.CHECKPOINT_DIR}USPS_teacher_da.pth',
                       map_location=config.DEVICE)
    usps_feat = FeatureExtractor(in_channels=1).to(config.DEVICE)
    usps_feat.load_state_dict(ckpt['feat_ext'])
    usps_clf = Classifier(num_classes=10).to(config.DEVICE)
    usps_clf.load_state_dict(ckpt['classifier'])

    # Generate DIs from USPS teacher
    usps_dis, usps_labels = generate_dis_from_teacher(
        usps_feat, usps_clf, 'USPS', num_per_class=2400
    )

    # ADDA: USPS → MNIST
    mnist_train_loader, mnist_test_loader = get_mnist_loaders()

    usps_to_mnist = adda_adapt(
        usps_feat, usps_clf,
        usps_dis, mnist_train_loader, mnist_test_loader,
        'USPS', 'MNIST', epochs=100
    )
    results['USPS→MNIST'] = usps_to_mnist

    # ═══════════════════════════════════
    # FINAL RESULTS
    # ═══════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TABLE 7 — DOMAIN ADAPTATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Adaptation':<20s} {'Ours':>8s}  {'Paper':>8s}")
    print(f"  {'-'*40}")
    print(f"  {'SVHN → MNIST':<20s} {results['SVHN→MNIST']:>7.2f}%  {'73.70%':>8s}")
    print(f"  {'MNIST → USPS':<20s} {results['MNIST→USPS']:>7.2f}%  {'82.20%':>8s}")
    print(f"  {'USPS → MNIST':<20s} {results['USPS→MNIST']:>7.2f}%  {'77.90%':>8s}")
    print(f"{'='*60}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    tasks = list(results.keys())
    ours = [results[t] for t in tasks]
    paper = [73.70, 82.20, 77.90]

    x = np.arange(len(tasks))
    width = 0.35
    ax.bar(x - width/2, ours, width, label='Ours', color='steelblue')
    ax.bar(x + width/2, paper, width, label='Paper', color='coral')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Domain Adaptation — ADDA with Data Impressions')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}domain_adaptation.png', dpi=150)
    plt.close()
    print(f"  ✅ Plot saved!")