import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import matplotlib.pyplot as plt

from models.alexnet import AlexNet, AlexNetHalf
from config import config

# ═══════════════════════════════════
# DI GENERATION WITH BATCHNORM HOOKS
# ═══════════════════════════════════

def sample_dirichlet(alpha, beta, n):
    a = np.clip(alpha * beta, 1e-3, None)
    return torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)

def generate_di_batch_bn(model, targets, dev, bn_layers, lambda_bn=10.0, steps=100):
    B = targets.shape[0]
    di = torch.randn(B, 3, 32, 32, device=dev)
    di.requires_grad_(True)
    targets = targets.to(dev)
    opt = torch.optim.Adam([di], lr=0.01)
    
    # Hook activations setup
    feature_activations = []
    def make_hook():
        def hook(module, input, output):
            feature_activations.append(output)
        return hook
        
    hooks = []
    for layer in bn_layers:
        hooks.append(layer.register_forward_hook(make_hook()))
        
    for _ in range(steps):
        opt.zero_grad()
        feature_activations.clear()
        
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        loss_dir = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        
        # BN Loss
        loss_bn = 0
        if len(bn_layers) > 0 and len(feature_activations) > 0:
            for act, bn in zip(feature_activations, bn_layers):
                mean_val = act.mean(dim=[0, 2, 3])
                var_val = act.var(dim=[0, 2, 3], unbiased=False)
                loss_bn += torch.mean((mean_val - bn.running_mean) ** 2)
                loss_bn += torch.mean((var_val - bn.running_var) ** 2)
            
        total_loss = loss_dir + lambda_bn * loss_bn
        total_loss.backward()
        opt.step()
        
        with torch.no_grad():
            di.clamp_(-1.0, 1.0)
            
    for h in hooks:
        h.remove()
        
    return di.detach()

def compute_similarity(model):
    w = model.get_final_weights()
    sim = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim.cpu().numpy()

def generate_full_di(model, num_per_class, bn_layers, use_bn=True, lambda_bn=10.0):
    model.eval()
    sim = compute_similarity(model)
    num_classes = sim.shape[0]
    all_dis, all_labels = [], []
    
    for cls in range(num_classes):
        alpha = sim[cls]
        class_dis = []
        for beta in [1.0, 0.1]:
            n_per_beta = num_per_class // 2
            targets = sample_dirichlet(alpha, beta, n_per_beta)
            for start in range(0, n_per_beta, 500):
                end = min(start + 500, n_per_beta)
                batch_t = targets[start:end]
                if use_bn:
                    di = generate_di_batch_bn(model, batch_t, config.DEVICE, bn_layers, lambda_bn, 100)
                else:
                    di = generate_di_batch_bn(model, batch_t, config.DEVICE, [], 0.0, 100)
                class_dis.append(di.cpu())
        ct = torch.cat(class_dis, dim=0)
        all_dis.append(ct)
        all_labels.extend([cls] * num_per_class)
        
    return torch.cat(all_dis, dim=0), torch.tensor(all_labels, dtype=torch.long)

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

def train_student_eval(teacher, student, di_loader, test_loader, epochs=30):
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    best_acc = 0
    for epoch in range(1, epochs + 1):
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
    return best_acc

def run_bn_extension_experiments():
    print("="*60)
    print("  EXTENSION 1: BATCH NORMALIZATION STATS MATCHING")
    print("="*60)
    
    # ── CIFAR-10 (AlexNet) ──
    print("\n>>> Running CIFAR-10 Experiment...")
    teacher = AlexNet().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
    teacher.eval()
    
    bn_layers = [m for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)]
    print(f"  Found {len(bn_layers)} BatchNorm layers in Teacher features.")
    
    # Generate DIs without BN (standard)
    print("  Generating CIFAR-10 DIs without BN regularization...")
    dis_no_bn, labels_no_bn = generate_full_di(teacher, 4000, [], use_bn=False)
    
    # Generate DIs with BN
    print("  Generating CIFAR-10 DIs WITH BN regularization...")
    dis_bn, labels_bn = generate_full_di(teacher, 4000, bn_layers, use_bn=True, lambda_bn=10.0)
    
    # Save a visualization of the generated DIs
    plt.figure(figsize=(10, 5))
    # Denormalize for plotting (mean=0.5, std=0.5 was used, so x_plot = x * 0.5 + 0.5)
    for i in range(5):
        img_no_bn = (dis_no_bn[i*4000].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        plt.subplot(2, 5, i+1)
        plt.imshow(img_no_bn)
        plt.axis('off')
        if i == 2: plt.title("Standard DIs (No BN)")
        
        img_bn = (dis_bn[i*4000].permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        plt.subplot(2, 5, i+6)
        plt.imshow(img_bn)
        plt.axis('off')
        if i == 2: plt.title("Realism DIs (With BN)")
    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{config.RESULTS_DIR}di_bn_realism.png', dpi=150)
    plt.close()
    print("  Visualizations saved: results/di_bn_realism.png")
    
    # Evaluate Student training
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    test_data = datasets.CIFAR10('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Train Student No BN
    print("  Training student on standard DIs...")
    student_no_bn = AlexNetHalf().to(config.DEVICE)
    ds_no_bn = AugDIDataset(dis_no_bn, labels_no_bn)
    dl_no_bn = DataLoader(ds_no_bn, batch_size=128, shuffle=True)
    acc_no_bn = train_student_eval(teacher, student_no_bn, dl_no_bn, test_loader, epochs=200)
    
    # Train Student WITH BN
    print("  Training student on BN regularized DIs...")
    student_bn = AlexNetHalf().to(config.DEVICE)
    ds_bn = AugDIDataset(dis_bn, labels_bn)
    dl_bn = DataLoader(ds_bn, batch_size=128, shuffle=True)
    acc_bn = train_student_eval(teacher, student_bn, dl_bn, test_loader, epochs=200)
    
    print("\n" + "="*50)
    print("  CIFAR-10 EXTENSION RESULTS")
    print("="*50)
    print(f"  Standard ZSKD (No BN): {acc_no_bn:.2f}%")
    print(f"  BN Statistics ZSKD   : {acc_bn:.2f}%")
    print(f"  Absolute Improvement : {acc_bn - acc_no_bn:+.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_bn_extension_experiments()
