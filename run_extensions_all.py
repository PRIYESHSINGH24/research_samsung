import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from models.alexnet import AlexNet, AlexNetHalf
from config import config

# ═══════════════════════════════════
# HELPER SAMPLERS & UTILITIES
# ═══════════════════════════════════

def sample_dirichlet(alpha, beta, n):
    """Standard Dirichlet target sampling (Base Method)"""
    a = np.clip(alpha * beta, 1e-3, None)
    return torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)

def sample_lognormal(alpha, beta, n):
    """
    EXTENSION 3: Dirichlet Replacement (Log-Normal target sampler)
    Instead of Dirichlet, model target logits as Log-Normal distributed.
    """
    num_classes = len(alpha)
    # Convert alpha (similarities) to logit space base means
    alpha_clipped = np.clip(alpha, 1e-6, 1 - 1e-6)
    base_mu = np.log(alpha_clipped / (1.0 - alpha_clipped))
    
    # Scale base_mu and use beta to control variance in log-normal/logit space
    # High concentration (beta=1.0) -> lower variance. Low concentration (beta=0.1) -> higher variance.
    variance = 1.0 / (beta + 1e-4)
    
    targets_list = []
    for _ in range(n):
        # Sample logits from Gaussian
        logits = np.random.normal(loc=base_mu, scale=np.sqrt(variance))
        # Apply Softmax to get target probability vector
        prob = np.exp(logits) / np.sum(np.exp(logits))
        targets_list.append(prob)
        
    return torch.tensor(np.array(targets_list), dtype=torch.float32)

# ═══════════════════════════════════
# MULTI-LAYER FEATURE EXTRACTION (EXTENSION 2)
# ═══════════════════════════════════

class AlexNetMultiLayer(nn.Module):
    """Hook wrapper to extract intermediate features from AlexNet"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.features = {}

    def forward(self, x, temperature=1.0):
        # Extract features at each pool/activation layer
        feat = x
        pool_count = 0
        self.features.clear()
        
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
# DATA IMPRESSIONS GENERATION CORE
# ═══════════════════════════════════

def compute_similarity(model):
    """Cosine similarity of final layer weights (Base Method)"""
    w = model.get_final_weights()
    sim = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.cpu().numpy()

def compute_multilayer_similarity(teacher, stage1_loader):
    """
    EXTENSION 2: Multilayer Similarity matrix computation.
    """
    print("  Computing Multilayer Similarity Matrix across all convolutions/pools...")
    multi_model = AlexNetMultiLayer(teacher).to(config.DEVICE)
    multi_model.eval()
    
    num_classes = 10
    layer_features = {}
    
    with torch.no_grad():
        for images, labels in stage1_loader:
            images = images.to(config.DEVICE)
            all_feats = multi_model.get_all_features(images)
            
            for layer_name, feats in all_feats.items():
                if layer_name not in layer_features:
                    layer_features[layer_name] = {c: [] for c in range(num_classes)}
                if feats.dim() == 4:
                    feats = F.adaptive_avg_pool2d(feats, 1).squeeze(-1).squeeze(-1)
                for i, label in enumerate(labels):
                    layer_features[layer_name][label.item()].append(feats[i].cpu())
                    
    layer_means = {}
    for layer_name, class_feats in layer_features.items():
        means = []
        for c in range(num_classes):
            if len(class_feats[c]) > 0:
                means.append(torch.stack(class_feats[c]).mean(0))
            else:
                means.append(torch.zeros_like(list(class_feats.values())[0][0]))
        layer_means[layer_name] = torch.stack(means)
        
    layer_sims = {}
    for layer_name, means in layer_means.items():
        sim = F.cosine_similarity(means.unsqueeze(1), means.unsqueeze(0), dim=2)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        layer_sims[layer_name] = sim.numpy()
        
    # Increasing weights (deeper layers get more weight)
    names = list(layer_sims.keys())
    weights = {name: (i+1)/sum(range(1, len(names)+1)) for i, name in enumerate(names)}
    
    combined_sim = np.zeros((num_classes, num_classes))
    for layer_name, sim in layer_sims.items():
        combined_sim += weights[layer_name] * sim
        
    combined_sim = (combined_sim - combined_sim.min()) / (combined_sim.max() - combined_sim.min() + 1e-8)
    return combined_sim

def generate_di_batch(model, targets, dev, ext_prior=False, lambda_tv=1e-4, lambda_l2=1e-5):
    """
    Generate a batch of DIs with optional EXTENSION 4 (TV + L2 Priors)
    """
    B = targets.shape[0]
    di = torch.randn(B, 3, 32, 32, device=dev)
    di.requires_grad_(True)
    targets = targets.to(dev)
    opt = torch.optim.Adam([di], lr=0.1)
    
    for _ in range(400):  # 400 steps for super-full scale matching
        opt.zero_grad()
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        
        # Cross Entropy Loss
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        
        # EXTENSION 4: Add TV and L2 Priors
        if ext_prior:
            loss_tv = torch.sum(torch.abs(di[:, :, :, 1:] - di[:, :, :, :-1])) + \
                      torch.sum(torch.abs(di[:, :, 1:, :] - di[:, :, :-1, :]))
            loss_l2 = torch.mean(di ** 2)
            loss = loss + lambda_tv * loss_tv + lambda_l2 * loss_l2
            
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            di.clamp_(-1.0, 1.0)
            
    return di.detach()

def generate_full_di(model, num_per_class, sim_matrix, ext_sampler=False, ext_prior=False):
    """Generate all DIs for ZSKD"""
    model.eval()
    num_classes = sim_matrix.shape[0]
    all_dis, all_labels = [], []
    
    for cls in range(num_classes):
        alpha = sim_matrix[cls]
        class_dis = []
        for beta in [1.0, 0.1]:
            n_per_beta = num_per_class // 2
            
            # EXTENSION 3: Dirichlet Replacement
            if ext_sampler:
                targets = sample_lognormal(alpha, beta, n_per_beta)
            else:
                targets = sample_dirichlet(alpha, beta, n_per_beta)
                
            for start in range(0, n_per_beta, 500):
                end = min(start + 500, n_per_beta)
                batch_t = targets[start:end]
                di = generate_di_batch(model, batch_t, config.DEVICE, ext_prior=ext_prior)
                class_dis.append(di.cpu())
                
        ct = torch.cat(class_dis, dim=0)
        all_dis.append(ct)
        all_labels.extend([cls] * num_per_class)
        
    return torch.cat(all_dis, dim=0), torch.tensor(all_labels, dtype=torch.long)

# ═══════════════════════════════════
# STUDENT TRAINING & DYNAMIC SIMILARITY (EXTENSION 5)
# ═══════════════════════════════════

class AugDIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return self.images.shape[0]
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

def train_student(teacher, student, di_loader, test_loader, ext_dynamic=False, epochs=500):
    """
    Train student model.
    EXTENSION 5: Dynamic class similarity matching via target mixup based on student boundaries.
    """
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        student.train()
        for batch, _ in di_loader:
            batch = batch.to(config.DEVICE)
            with torch.no_grad():
                t_logits = teacher(batch)
            s_logits = student(batch)
            
            # Target softening
            soft_t = F.softmax(t_logits / 20, dim=1)
            
            # EXTENSION 5: Dynamic Similarity Mixing
            if ext_dynamic:
                # Calculate active prediction overlaps on student output
                with torch.no_grad():
                    soft_s = F.softmax(s_logits / 20, dim=1)
                # Mix target distributions to regularize drifting boundaries
                mix_coeff = 0.15 * (epoch / epochs)  # linearly increase regularization coeff
                soft_t = (1.0 - mix_coeff) * soft_t + mix_coeff * soft_s
                
            log_s = F.log_softmax(s_logits / 20, dim=1)
            loss = F.kl_div(log_s, soft_t, reduction='batchmean') * (20 ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval epoch
        if epoch % 50 == 0 or epoch == epochs:
            student.eval()
            correct = total = 0
            with torch.no_grad():
                for img, lab in test_loader:
                    img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                    _, pred = student(img).max(1)
                    total += lab.size(0)
                    correct += pred.eq(lab).sum().item()
            acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
            print(f"  Epoch {epoch}/{epochs} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
            
    return best_acc

# ═══════════════════════════════════
# MAIN EXECUTIVE PIPELINE
# ═══════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run Data Impressions Extensions on CIFAR-10")
    parser.add_argument("--extension", type=int, default=2, choices=[2, 3, 4, 5],
                        help="Select extension to run (2: Multilayer Sim, 3: Log-Normal Sampler, 4: TV+L2 Priors, 5: Dynamic Sim)")
    parser.add_argument("--num_di", type=int, default=40000, help="Total synthesized DIs (4000 per class)")
    parser.add_argument("--epochs", type=int, default=500, help="Student distillation training epochs")
    args = parser.parse_args()
    
    print("="*70)
    print(f"  RUNNING EXTENSION {args.extension} (Super-Full Scale ZSKD)")
    print(f"  Total DIs: {args.num_di} | Epochs: {args.epochs}")
    print("="*70)
    
    # Load CIFAR-10 datasets
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    test_loader = DataLoader(datasets.CIFAR10('data/', train=False, transform=tf_test), batch_size=256, shuffle=False)
    
    # Load Teacher model
    teacher = AlexNet().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
    teacher.eval()
    
    # ── BASELINE RUN: Standard ZSKD ──
    print("\n>>> Stage 0: Evaluating Baseline ZSKD...")
    base_sim = compute_similarity(teacher)
    
    # ── GENERATION & EVALUATION ──
    if args.extension == 2:
        print("\n>>> Running Extension 2: Multilayer Similarity Matrix...")
        # 1-Stage DIs to compute similarity
        print("  Generating Stage-1 standard DIs as proxy data...")
        s1_dis, s1_labels = generate_full_di(teacher, 2000, base_sim)
        s1_loader = DataLoader(AugDIDataset(s1_dis, s1_labels), batch_size=128, shuffle=False)
        # Compute Multilayer matrix
        multi_sim = compute_multilayer_similarity(teacher, s1_loader)
        # Generate Final DIs using Multilayer Sim
        print("  Generating final DIs using Multilayer Similarity...")
        dis_ext, labels_ext = generate_full_di(teacher, args.num_di // 10, multi_sim)
        
    elif args.extension == 3:
        print("\n>>> Running Extension 3: Log-Normal Target Sampling (Dirichlet Replacement)...")
        dis_ext, labels_ext = generate_full_di(teacher, args.num_di // 10, base_sim, ext_sampler=True)
        
    elif args.extension == 4:
        print("\n>>> Running Extension 4: TV & L2 Prior Regularization...")
        dis_ext, labels_ext = generate_full_di(teacher, args.num_di // 10, base_sim, ext_prior=True)
        
    elif args.extension == 5:
        print("\n>>> Running Extension 5: Dynamic Class Similarity Matrix...")
        # Uses standard DIs, but dynamic updates occur during student training
        dis_ext, labels_ext = generate_full_di(teacher, args.num_di // 10, base_sim)
        
    # Generate standard baseline DIs for parallel comparison if needed
    print("\n>>> Generating standard baseline DIs for comparison...")
    dis_base, labels_base = generate_full_di(teacher, args.num_di // 10, base_sim)
    
    # Train standard baseline student
    print("\n>>> Training student on standard DIs...")
    student_base = AlexNetHalf().to(config.DEVICE)
    dl_base = DataLoader(AugDIDataset(dis_base, labels_base), batch_size=128, shuffle=True)
    acc_base = train_student(teacher, student_base, dl_base, test_loader, epochs=args.epochs)
    
    # Train extension student
    print(f"\n>>> Training student on Extension {args.extension} DIs...")
    student_ext = AlexNetHalf().to(config.DEVICE)
    dl_ext = DataLoader(AugDIDataset(dis_ext, labels_ext), batch_size=128, shuffle=True)
    acc_ext = train_student(teacher, student_ext, dl_ext, test_loader, 
                            ext_dynamic=(args.extension == 5), epochs=args.epochs)
    
    # Output Scorecard
    print("\n" + "="*50)
    print(f"  ZSKD EXTENSION {args.extension} REPORT CARD")
    print("="*50)
    print(f"  Standard ZSKD Baseline: {acc_base:.2f}%")
    print(f"  Extension {args.extension} ZSKD   : {acc_ext:.2f}%")
    print(f"  Absolute Improvement  : {acc_ext - acc_base:+.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
