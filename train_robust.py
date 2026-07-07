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
# ATTACKS
# ═══════════════════════════════════

def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    return perturbed.detach()


def ifgsm_attack(model, images, labels, epsilon, alpha=None, steps=10):
    if alpha is None:
        alpha = epsilon / steps * 2
    perturbed = images.clone().detach()
    
    for _ in range(steps):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        perturbed = perturbed + alpha * perturbed.grad.sign()
        # Project back to epsilon ball
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = (images + delta).detach()
    
    return perturbed


def pgd_attack(model, images, labels, epsilon, alpha=None, steps=20):
    if alpha is None:
        alpha = epsilon / steps * 2.5
    
    # Random start
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    perturbed = (images + delta).detach()
    
    for _ in range(steps):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        perturbed = perturbed + alpha * perturbed.grad.sign()
        delta = torch.clamp(perturbed - images, -epsilon, epsilon)
        perturbed = (images + delta).detach()
    
    return perturbed


# ═══════════════════════════════════
# DATA
# ═══════════════════════════════════

def get_loaders():
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
        train_data, batch_size=128,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=128,
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


def evaluate_attack(model, loader, attack_fn, epsilon):
    model.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        perturbed = attack_fn(model, images, labels, epsilon)
        
        with torch.no_grad():
            out = model(perturbed)
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


# ═══════════════════════════════════
# STEP 1: ADVERSARIAL TEACHER TRAINING
# ═══════════════════════════════════

def train_robust_teacher(epochs=100, pgd_epsilon=0.3):
    print("\n" + "="*50)
    print("  STEP 1: ADVERSARIAL TEACHER TRAINING")
    print("  Method: PGD Adversarial Training")
    print(f"  PGD Epsilon: {pgd_epsilon}")
    print("="*50)
    
    train_loader, test_loader = get_loaders()
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
            
            # Generate PGD adversarial examples
            teacher.eval()
            adv_images = pgd_attack(
                teacher, images, labels,
                epsilon=pgd_epsilon, steps=7
            )
            teacher.train()
            
            # Train on adversarial examples
            optimizer.zero_grad()
            outputs = teacher(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Test clean accuracy
        test_acc = evaluate(teacher, test_loader)
        train_acc = 100. * correct / total
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                teacher.state_dict(),
                f'{config.CHECKPOINT_DIR}robust_teacher.pth'
            )
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] "
                  f"Train: {train_acc:.2f}% | "
                  f"Test: {test_acc:.2f}%")
    
    print(f"\n  Robust Teacher Best: {best_acc:.2f}%")
    return teacher, best_acc


# ═══════════════════════════════════
# STEP 2: GENERATE DIs FROM ROBUST TEACHER
# ═══════════════════════════════════

def generate_robust_dis(teacher, num_per_class=2400):
    print("\n" + "="*50)
    print("  STEP 2: GENERATE DIs FROM ROBUST TEACHER")
    print("="*50)
    
    teacher.eval()
    sim_matrix = compute_similarity_matrix(teacher)
    
    all_images = []
    all_labels = []
    n_per_beta = num_per_class // len(config.BETA_VALUES)
    
    for class_idx in range(config.NUM_CLASSES):
        alpha_k = sim_matrix[class_idx]
        class_dis = []
        
        for beta in config.BETA_VALUES:
            sampled = sample_dirichlet_vectors(alpha_k, beta, n_per_beta)
            
            pbar = tqdm(
                range(0, n_per_beta, config.DI_BATCH_SIZE),
                desc=f"  Class {class_idx} β={beta}"
            )
            for start in pbar:
                end = min(start + config.DI_BATCH_SIZE, n_per_beta)
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
    }, f'{config.CHECKPOINT_DIR}robust_dis.pth')
    
    print(f"  Total: {all_images.shape}")
    print(f"  ✅ Saved!")
    return all_images, all_labels


# ═══════════════════════════════════
# STEP 3: KD — STUDENT FROM ROBUST DIs
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


def train_robust_student(teacher, di_images, di_labels, epochs=300):
    print("\n" + "="*50)
    print("  STEP 3: KD — STUDENT FROM ROBUST DIs")
    print("="*50)
    
    _, test_loader = get_loaders()
    
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
    temperature = config.KD_TEMPERATURE
    
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
            torch.save(
                student.state_dict(),
                f'{config.CHECKPOINT_DIR}robust_student.pth'
            )
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}] Test: {test_acc:.2f}% | Best: {best_acc:.2f}%")
    
    print(f"\n  Student Best: {best_acc:.2f}%")
    return student, best_acc


# ═══════════════════════════════════
# STEP 4: EVALUATE ROBUSTNESS
# ═══════════════════════════════════

def evaluate_robustness(teacher, student):
    print("\n" + "="*50)
    print("  STEP 4: ADVERSARIAL ROBUSTNESS EVALUATION")
    print("="*50)
    
    _, test_loader = get_loaders()
    
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    attacks = {
        'FGSM': fgsm_attack,
        'iFGSM': lambda m, i, l, e: ifgsm_attack(m, i, l, e, steps=10),
        'PGD': lambda m, i, l, e: pgd_attack(m, i, l, e, steps=20),
    }
    
    # Clean accuracy
    teacher_clean = evaluate(teacher, test_loader)
    student_clean = evaluate(student, test_loader)
    
    print(f"\n  Clean Accuracy:")
    print(f"    Teacher: {teacher_clean:.2f}%")
    print(f"    Student: {student_clean:.2f}%")
    
    results = {}
    
    for attack_name, attack_fn in attacks.items():
        print(f"\n  --- {attack_name} Attack ---")
        results[attack_name] = {'teacher': [], 'student': []}
        
        for eps in epsilons:
            t_acc = evaluate_attack(teacher, test_loader, attack_fn, eps)
            s_acc = evaluate_attack(student, test_loader, attack_fn, eps)
            
            results[attack_name]['teacher'].append(t_acc)
            results[attack_name]['student'].append(s_acc)
            
            print(f"  ε={eps:.2f}: Teacher={t_acc:.2f}% | Student={s_acc:.2f}%")
    
    # Print final table
    print(f"\n{'='*60}")
    print(f"  TABLE 4 — ADVERSARIAL ROBUSTNESS RESULTS")
    print(f"{'='*60}")
    print(f"  {'Attack':<8s} {'ε':<6s} {'Teacher':>10s} {'Student':>10s}")
    print(f"  {'-'*36}")
    print(f"  {'Clean':<8s} {'—':<6s} {teacher_clean:>9.2f}% {student_clean:>9.2f}%")
    
    for attack_name in attacks:
        for i, eps in enumerate(epsilons):
            t = results[attack_name]['teacher'][i]
            s = results[attack_name]['student'][i]
            print(f"  {attack_name:<8s} {eps:<6.2f} {t:>9.2f}% {s:>9.2f}%")
    print(f"{'='*60}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, attack_name in enumerate(attacks):
        ax = axes[idx]
        ax.plot(epsilons, results[attack_name]['teacher'], 'bo-',
                label='Robust Teacher', linewidth=2)
        ax.plot(epsilons, results[attack_name]['student'], 'rs--',
                label='ZSKD Student', linewidth=2)
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{attack_name} Attack')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle('Adversarial Robustness — MNIST')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}adversarial_robustness.png', dpi=150)
    plt.close()
    print(f"  ✅ Plot saved!")
    
    return results


# ═══════════════════════════════════
# MAIN
# ═══════════════════════════════════

if __name__ == "__main__":
    print("="*50)
    print("  SECTION 4.1.7 — ADVERSARIAL ROBUSTNESS")
    print("="*50)
    
    # Step 1: Train robust teacher
    teacher, t_acc = train_robust_teacher(epochs=100, pgd_epsilon=0.3)
    
    # Load best
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}robust_teacher.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()
    
    # Step 2: Generate DIs
    dis, labels = generate_robust_dis(teacher, num_per_class=2400)
    
    # Step 3: Train student
    student, s_acc = train_robust_student(teacher, dis, labels, epochs=300)
    
    # Load best student
    student = LeNet5Half().to(config.DEVICE)
    student.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}robust_student.pth',
            map_location=config.DEVICE
        )
    )
    
    # Step 4: Evaluate robustness
    results = evaluate_robustness(teacher, student)