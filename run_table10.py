"""
TABLE 10 — UAP Generation (Section 4.4)
Paper Settings:
  - Generator: 4 deconv layers, tanh×ε
  - ε=10 in [0,255] → scaled to signal range
  - Latent: 10-dim from U[-1,1], batch=32
  - 20 epochs, Adam
  - CIFAR: α=3e-04, lr=1e-05 (AlexNet)
  - MNIST/FMNIST: α=1e-04, lr=1e-05 (LeNet)
  - Fooling Loss + Diversity Loss from [35]
  - Compare CI vs DI
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np, os
from models.lenet import LeNet5
from models.alexnet import AlexNet
from models.generator import UAPGenerator  # Need to verify this exists
from config import config


# ═══════════════════════════════════
# UAP GENERATOR (Paper: 4 deconv, tanh×ε)
# ═══════════════════════════════════

class UAPGen(nn.Module):
    """Paper Section 4.4: 4 deconvolutional layers, tanh×ε"""
    def __init__(self, latent_dim=10, channels=1, epsilon=10/255):
        super().__init__()
        self.epsilon = epsilon
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z) * self.epsilon


# ═══════════════════════════════════
# DI/CI GENERATION
# ═══════════════════════════════════

def compute_sim(model):
    w = model.get_final_weights()
    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    return ((s - s.min()) / (s.max() - s.min() + 1e-8)).cpu().numpy()

def gen_di_batch(model, targets, dev, ch=1):
    B = targets.shape[0]; di = torch.randn(B,ch,32,32,device=dev); di.requires_grad_(True)
    targets = targets.to(dev); opt = torch.optim.Adam([di], lr=0.1)
    for _ in range(1500):
        opt.zero_grad(); logits = model(di, temperature=20); pred = F.softmax(logits,1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward(); opt.step()
        with torch.no_grad(): di.clamp_(-2.5, 2.5)
    return di.detach()

def generate_dis(model, n_per_class, ch=1):
    """DIs with β={1.0, 0.1}"""
    sim = compute_sim(model); all_dis = []
    for c in range(10):
        cdis = []
        for beta in [1.0, 0.1]:
            n = n_per_class // 2; a = np.clip(sim[c]*beta, 1e-3, None)
            tgts = torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)
            for s in tqdm(range(0,n,32), desc=f"  DI C{c} β={beta}"):
                e = min(s+32,n); cdis.append(gen_di_batch(model, tgts[s:e], config.DEVICE, ch).cpu())
        all_dis.append(torch.cat(cdis, 0))
        print(f"  ✅ DI C{c}")
    return torch.cat(all_dis, 0)

def generate_cis(model, n_per_class, ch=1):
    """Class Impressions: one-hot targets"""
    all_cis = []
    for c in range(10):
        target = torch.zeros(1, 10); target[0, c] = 1.0
        targets = target.repeat(n_per_class, 1)
        cdis = []
        for s in tqdm(range(0, n_per_class, 32), desc=f"  CI C{c}"):
            e = min(s+32, n_per_class)
            cdis.append(gen_di_batch(model, targets[s:e], config.DEVICE, ch).cpu())
        all_cis.append(torch.cat(cdis, 0))
    return torch.cat(all_cis, 0)


# ═══════════════════════════════════
# UAP TRAINING (Fooling + Diversity Loss)
# ═══════════════════════════════════

def fooling_loss(classifier, images, uap):
    """Minimize confidence on predicted class"""
    with torch.no_grad():
        clean_out = classifier(images); _, clean_pred = clean_out.max(1)
    adv_out = classifier(images + uap)
    # Maximize cross-entropy = minimize confidence on original prediction
    return -nn.CrossEntropyLoss()(adv_out, clean_pred)

def diversity_loss(uaps):
    """Encourage diverse UAPs"""
    B = uaps.size(0)
    uaps_flat = uaps.view(B, -1)
    norm = uaps_flat.norm(dim=1, keepdim=True) + 1e-8
    uaps_norm = uaps_flat / norm
    sim = torch.mm(uaps_norm, uaps_norm.t())
    mask = ~torch.eye(B, dtype=torch.bool, device=uaps.device)
    return sim[mask].mean()

def train_uap(classifier, generator, train_images, epochs, lr, alpha_div, name):
    """Paper: 20 epochs, batch=32, Adam"""
    print(f"\n  Training UAP: {name}")
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    n = len(train_images)

    for ep in range(1, epochs+1):
        generator.train()
        perm = torch.randperm(n)
        total_loss = 0; batches = 0
        for start in range(0, n, 32):
            end = min(start+32, n)
            idx = perm[start:end]
            images = train_images[idx].to(config.DEVICE)

            z = torch.empty(images.size(0), 10).uniform_(-1, 1).to(config.DEVICE)
            uap = generator(z)
            if uap.size(2) != images.size(2):
                uap = F.interpolate(uap, size=images.shape[2:])

            f_loss = fooling_loss(classifier, images, uap)
            d_loss = diversity_loss(uap)
            loss = f_loss + alpha_div * d_loss

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); batches += 1

        if ep % 5 == 0:
            print(f"    Ep {ep}: Loss={total_loss/batches:.4f}")

    return generator


def eval_fooling_rate(classifier, generator, test_loader):
    """Compute fooling rate on test set"""
    classifier.eval(); generator.eval()
    fooled = total = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img = img.to(config.DEVICE)
            z = torch.empty(img.size(0), 10).uniform_(-1, 1).to(config.DEVICE)
            uap = generator(z)
            if uap.size(2) != img.size(2):
                uap = F.interpolate(uap, size=img.shape[2:])
            _, clean_pred = classifier(img).max(1)
            _, adv_pred = classifier(img + uap).max(1)
            total += img.size(0)
            fooled += (clean_pred != adv_pred).sum().item()
    return 100.*fooled/total


def evaluate(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            _, p = model(img).max(1); t += lab.size(0); c += p.eq(lab).sum().item()
    return 100.*c/t


if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print("="*60)
    print("  TABLE 10 — UAP GENERATION (ALL 3 DATASETS)")
    print("  CI vs DI comparison")
    print("="*60)

    results = {}

    # ── MNIST ──
    print("\n  === MNIST (LeNet) ===")
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_data = datasets.MNIST('data/', train=True, download=True, transform=tf)
    test_data = datasets.MNIST('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}mnist_teacher.pth', map_location=config.DEVICE))
    teacher.eval()
    eps = 10/255

    # CI UAP
    print("  Generating CIs...")
    cis = generate_cis(teacher, 240, ch=1)  # 2400 total
    gen_ci = UAPGen(10, 1, eps).to(config.DEVICE)
    train_uap(teacher, gen_ci, cis, 20, 1e-5, 1e-4, "MNIST CI")
    ci_fr = eval_fooling_rate(teacher, gen_ci, test_loader)

    # DI UAP
    print("  Generating DIs...")
    dis = generate_dis(teacher, 240, ch=1)  # 2400 total
    gen_di = UAPGen(10, 1, eps).to(config.DEVICE)
    train_uap(teacher, gen_di, dis, 20, 1e-5, 1e-4, "MNIST DI")
    di_fr = eval_fooling_rate(teacher, gen_di, test_loader)
    results['mnist'] = (ci_fr, di_fr)
    print(f"  MNIST: CI={ci_fr:.2f}% DI={di_fr:.2f}%")

    # ── FMNIST ──
    print("\n  === FMNIST (LeNet) ===")
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.2860,),(0.3530,))])
    train_data = datasets.FashionMNIST('data/', train=True, download=True, transform=tf)
    test_data = datasets.FashionMNIST('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth', map_location=config.DEVICE))
    teacher.eval()

    cis = generate_cis(teacher, 240, ch=1)
    gen_ci = UAPGen(10, 1, eps).to(config.DEVICE)
    train_uap(teacher, gen_ci, cis, 20, 1e-5, 1e-4, "FMNIST CI")
    ci_fr = eval_fooling_rate(teacher, gen_ci, test_loader)

    dis = generate_dis(teacher, 240, ch=1)
    gen_di = UAPGen(10, 1, eps).to(config.DEVICE)
    train_uap(teacher, gen_di, dis, 20, 1e-5, 1e-4, "FMNIST DI")
    di_fr = eval_fooling_rate(teacher, gen_di, test_loader)
    results['fmnist'] = (ci_fr, di_fr)
    print(f"  FMNIST: CI={ci_fr:.2f}% DI={di_fr:.2f}%")

    # ── CIFAR ──
    print("\n  === CIFAR-10 (AlexNet) ===")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    test_data = datasets.CIFAR10('data/', train=False, transform=tf)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)

    teacher = AlexNet().to(config.DEVICE)
    teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
    teacher.eval()

    cis = generate_cis(teacher, 240, ch=3)
    gen_ci = UAPGen(10, 3, eps).to(config.DEVICE)
    train_uap(teacher, gen_ci, cis, 20, 1e-5, 3e-4, "CIFAR CI")
    ci_fr = eval_fooling_rate(teacher, gen_ci, test_loader)

    dis = generate_dis(teacher, 240, ch=3)
    gen_di = UAPGen(10, 3, eps).to(config.DEVICE)
    train_uap(teacher, gen_di, dis, 20, 1e-5, 3e-4, "CIFAR DI")
    di_fr = eval_fooling_rate(teacher, gen_di, test_loader)
    results['cifar'] = (ci_fr, di_fr)
    print(f"  CIFAR: CI={ci_fr:.2f}% DI={di_fr:.2f}%")

    # ── FINAL TABLE ──
    print(f"\n{'='*60}")
    print(f"  TABLE 10 — UAP FOOLING RATES")
    print(f"{'='*60}")
    print(f"  {'Method':<12s} {'AlexNet(CIFAR)':>14s} {'LeNet(FMNIST)':>14s} {'LeNet(MNIST)':>14s}")
    print(f"  {'-'*55}")
    ci_c, ci_f, ci_m = results['cifar'][0], results['fmnist'][0], results['mnist'][0]
    di_c, di_f, di_m = results['cifar'][1], results['fmnist'][1], results['mnist'][1]
    print(f"  {'CI [35]':<12s} {ci_c:>13.2f}% {ci_f:>13.2f}% {ci_m:>13.2f}%")
    print(f"  {'DI (Ours)':<12s} {di_c:>13.2f}% {di_f:>13.2f}% {di_m:>13.2f}%")    
    print(f"  {'-'*55}")
    print(f"  {'Paper CI':<12s} {'90.18%':>14s} {'91.29%':>14s} {'91.10%':>14s}")
    print(f"  {'Paper DI':<12s} {'94.23%':>14s} {'96.37%':>14s} {'96.45%':>14s}")
    print(f"{'='*60}")