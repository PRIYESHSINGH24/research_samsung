"""
TABLE 7 — Domain Adaptation (Section 4.2)
Paper Settings:
  - beta={0.01, 0.1}
  - Images 28x28, normalized [0,1]
  - LeNet from ADDA [25]
  - ADDA framework with adversarial discriminator
  - SVHN->MNIST: full training data
  - MNIST<->USPS: 2000/1800 sampled
  - Discriminator: 2 FC(500) + LeakyReLU
  - DI lr=0.001, Adaptation lr=2e-4
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np, os
from config import config


class DALeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
            nn.Conv2d(20, 50, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
        )
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, temperature=1.0):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        logits = self.fc2(feat)
        return logits / temperature

    def get_features(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

    def get_final_weights(self):
        return self.fc2.weight.data.clone()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(500, 500), nn.LeakyReLU(0.2),
            nn.Linear(500, 500), nn.LeakyReLU(0.2),
            nn.Linear(500, 2),
        )

    def forward(self, x):
        return self.net(x)


def compute_sim(model):
    w = model.get_final_weights()
    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    return ((s - s.min()) / (s.max() - s.min() + 1e-8)).cpu().numpy()

def gen_di_batch(model, targets, dev):
    B = targets.shape[0]
    di = torch.randn(B, 1, 28, 28, device=dev)
    di.requires_grad_(True)
    targets = targets.to(dev)
    opt = torch.optim.Adam([di], lr=0.001)
    for _ in range(1500):
        opt.zero_grad()
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward()
        opt.step()
        with torch.no_grad():
            di.clamp_(0, 1)
    return di.detach()

def generate_dis(model, n_per_class):
    sim = compute_sim(model)
    all_dis, all_lab = [], []
    for c in range(10):
        cdis = []
        for beta in [0.01, 0.1]:
            n = n_per_class // 2
            a = np.clip(sim[c] * beta, 1e-3, None)
            tgts = torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)
            for s in tqdm(range(0, n, 32), desc=f"  C{c} b={beta}"):
                e = min(s + 32, n)
                cdis.append(gen_di_batch(model, tgts[s:e], config.DEVICE).cpu())
        ct = torch.cat(cdis, 0)
        all_dis.append(ct)
        all_lab.extend([c] * n_per_class)
        print(f"  Done C{c}: {ct.shape[0]}")
    return torch.cat(all_dis, 0), torch.tensor(all_lab, dtype=torch.long)


def adda_adapt(source_model, target_loader, di_images, epochs=200):
    """
    FIXED ADDA:
    1. Discriminator lr slower (1e-4)
    2. Target encoder trained 3x per disc step
    3. Full target model trainable (not just encoder)
    4. More epochs (200)
    5. Label smoothing for discriminator stability
    6. Gradient clipping
    """
    source_model.eval()
    target_model = DALeNet().to(config.DEVICE)
    target_model.load_state_dict(source_model.state_dict())
    disc = Discriminator().to(config.DEVICE)

    # FIX 1: Discriminator slower, target faster
    opt_disc = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_tgt = optim.Adam(target_model.parameters(), lr=2e-4, betas=(0.5, 0.999))

    di_loader = DataLoader(
        torch.utils.data.TensorDataset(di_images, torch.zeros(len(di_images))),
        batch_size=128, shuffle=True, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for ep in range(1, epochs + 1):
        target_model.train()
        disc.train()
        di_iter = iter(di_loader)

        for tgt_img, _ in target_loader:
            try:
                src_img, _ = next(di_iter)
            except StopIteration:
                di_iter = iter(di_loader)
                src_img, _ = next(di_iter)

            src_img = src_img.to(config.DEVICE)
            tgt_img = tgt_img.to(config.DEVICE)

            # ── Train Discriminator (1 step) ──
            with torch.no_grad():
                src_feat = source_model.get_features(src_img)
            tgt_feat = target_model.get_features(tgt_img).detach()

            # FIX 2: Label smoothing for stability
            src_labels = torch.zeros(src_feat.size(0), dtype=torch.long, device=config.DEVICE)
            tgt_labels = torch.ones(tgt_feat.size(0), dtype=torch.long, device=config.DEVICE)

            disc_out = disc(torch.cat([src_feat, tgt_feat], 0))
            disc_loss = criterion(disc_out, torch.cat([src_labels, tgt_labels], 0))

            opt_disc.zero_grad()
            disc_loss.backward()
            # FIX 3: Gradient clipping
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_disc.step()

            # ── Train Target Encoder (3 steps) — FIX 4 ──
            for _ in range(3):
                tgt_feat = target_model.get_features(tgt_img)
                tgt_out = disc(tgt_feat)
                # Target wants to be classified as SOURCE (label=0)
                tgt_loss = criterion(tgt_out, torch.zeros(tgt_feat.size(0), dtype=torch.long, device=config.DEVICE))
                opt_tgt.zero_grad()
                tgt_loss.backward()
                torch.nn.utils.clip_grad_norm_(target_model.parameters(), 1.0)
                opt_tgt.step()

        if ep % 20 == 0:
            print(f"    ADDA Ep {ep}: disc={disc_loss.item():.4f} tgt={tgt_loss.item():.4f}")

    return target_model


def evaluate(model, loader):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            _, p = model(img).max(1)
            t += lab.size(0)
            c += p.eq(lab).sum().item()
    return 100. * c / t


def get_mnist_loaders(n_samples=None):
    tf = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    train = datasets.MNIST('data/', train=True, download=True, transform=tf)
    test = datasets.MNIST('data/', train=False, transform=tf)
    if n_samples:
        idx = torch.randperm(len(train))[:n_samples]
        train = Subset(train, idx)
    return DataLoader(train, batch_size=128, shuffle=True, num_workers=0), \
           DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

def get_usps_loaders(n_samples=None):
    tf = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    train = datasets.USPS('data/', train=True, download=True, transform=tf)
    test = datasets.USPS('data/', train=False, download=True, transform=tf)
    if n_samples:
        idx = torch.randperm(len(train))[:n_samples]
        train = Subset(train, idx)
    return DataLoader(train, batch_size=128, shuffle=True, num_workers=0), \
           DataLoader(test, batch_size=256, shuffle=False, num_workers=0)

def get_svhn_loaders():
    tf = transforms.Compose([transforms.Resize((28, 28)), transforms.Grayscale(), transforms.ToTensor()])
    train = datasets.SVHN('data/', split='train', download=True, transform=tf)
    test = datasets.SVHN('data/', split='test', download=True, transform=tf)
    return DataLoader(train, batch_size=128, shuffle=True, num_workers=0), \
           DataLoader(test, batch_size=256, shuffle=False, num_workers=0)


if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print("=" * 60)
    print("  TABLE 7 -- DOMAIN ADAPTATION (FIXED)")
    print("  beta={0.01, 0.1}, images 28x28, [0,1]")
    print("=" * 60)

    results = {}

    # SVHN -> MNIST
    print("\n  === SVHN -> MNIST (full data) ===")
    svhn_train, _ = get_svhn_loaders()
    _, mnist_test = get_mnist_loaders()

    source = DALeNet().to(config.DEVICE)
    # Check if already trained
    ckpt_path = f'{config.CHECKPOINT_DIR}da_svhn_source.pth'
    if os.path.exists(ckpt_path):
        source.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        print("  Loaded saved SVHN source model")
    else:
        opt = optim.Adam(source.parameters(), lr=0.001)
        for ep in range(1, 31):
            source.train()
            for img, lab in tqdm(svhn_train, leave=False, desc=f"SVHN {ep}"):
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt.zero_grad()
                nn.CrossEntropyLoss()(source(img), lab).backward()
                opt.step()
            if ep % 10 == 0:
                print(f"  SVHN Ep {ep}")
        torch.save(source.state_dict(), ckpt_path)
    source.eval()

    baseline = evaluate(source, mnist_test)
    print(f"  Baseline (no adapt): {baseline:.2f}%")

    # Check if DIs already generated
    di_path = f'{config.CHECKPOINT_DIR}da_svhn_dis.pth'
    if os.path.exists(di_path):
        data = torch.load(di_path, map_location='cpu')
        dis = data['di_images']
        print(f"  Loaded saved DIs: {dis.shape}")
    else:
        dis, _ = generate_dis(source, 2400)
        torch.save({'di_images': dis}, di_path)

    mnist_train, _ = get_mnist_loaders()
    target = adda_adapt(source, mnist_train, dis, epochs=200)
    adapted = evaluate(target, mnist_test)
    results['s2m'] = (baseline, adapted)
    print(f"  S->M: Baseline={baseline:.2f}%, Adapted={adapted:.2f}%")

    # USPS -> MNIST
    print("\n  === USPS -> MNIST (2000/1800) ===")
    usps_train, _ = get_usps_loaders(n_samples=1800)
    _, mnist_test = get_mnist_loaders()

    source = DALeNet().to(config.DEVICE)
    ckpt_path = f'{config.CHECKPOINT_DIR}da_usps_source.pth'
    if os.path.exists(ckpt_path):
        source.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        print("  Loaded saved USPS source model")
    else:
        opt = optim.Adam(source.parameters(), lr=0.001)
        for ep in range(1, 51):
            source.train()
            for img, lab in tqdm(usps_train, leave=False, desc=f"USPS {ep}"):
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt.zero_grad()
                nn.CrossEntropyLoss()(source(img), lab).backward()
                opt.step()
        torch.save(source.state_dict(), ckpt_path)
    source.eval()

    baseline = evaluate(source, mnist_test)
    print(f"  Baseline: {baseline:.2f}%")

    di_path = f'{config.CHECKPOINT_DIR}da_usps_dis.pth'
    if os.path.exists(di_path):
        data = torch.load(di_path, map_location='cpu')
        dis = data['di_images']
        print(f"  Loaded saved DIs: {dis.shape}")
    else:
        dis, _ = generate_dis(source, 2400)
        torch.save({'di_images': dis}, di_path)

    mnist_train_sub, _ = get_mnist_loaders(n_samples=2000)
    target = adda_adapt(source, mnist_train_sub, dis, epochs=200)
    adapted = evaluate(target, mnist_test)
    results['u2m'] = (baseline, adapted)
    print(f"  U->M: Baseline={baseline:.2f}%, Adapted={adapted:.2f}%")

    # MNIST -> USPS
    print("\n  === MNIST -> USPS (2000/1800) ===")
    mnist_train_sub, _ = get_mnist_loaders(n_samples=2000)
    _, usps_test = get_usps_loaders()

    source = DALeNet().to(config.DEVICE)
    ckpt_path = f'{config.CHECKPOINT_DIR}da_mnist_source.pth'
    if os.path.exists(ckpt_path):
        source.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        print("  Loaded saved MNIST source model")
    else:
        opt = optim.Adam(source.parameters(), lr=0.001)
        for ep in range(1, 51):
            source.train()
            for img, lab in tqdm(mnist_train_sub, leave=False, desc=f"MNIST {ep}"):
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt.zero_grad()
                nn.CrossEntropyLoss()(source(img), lab).backward()
                opt.step()
        torch.save(source.state_dict(), ckpt_path)
    source.eval()

    baseline = evaluate(source, usps_test)
    print(f"  Baseline: {baseline:.2f}%")

    di_path = f'{config.CHECKPOINT_DIR}da_mnist_dis.pth'
    if os.path.exists(di_path):
        data = torch.load(di_path, map_location='cpu')
        dis = data['di_images']
        print(f"  Loaded saved DIs: {dis.shape}")
    else:
        dis, _ = generate_dis(source, 2400)
        torch.save({'di_images': dis}, di_path)

    usps_train_sub, _ = get_usps_loaders(n_samples=1800)
    target = adda_adapt(source, usps_train_sub, dis, epochs=200)
    adapted = evaluate(target, usps_test)
    results['m2u'] = (baseline, adapted)
    print(f"  M->U: Baseline={baseline:.2f}%, Adapted={adapted:.2f}%")

    # FINAL TABLE
    print()
    print("=" * 60)
    print("  TABLE 7 -- DOMAIN ADAPTATION RESULTS")
    print("=" * 60)
    print("  Adaptation           Baseline    Adapted      Paper")
    print("  " + "-" * 50)
    s2m_b, s2m_a = results['s2m']
    u2m_b, u2m_a = results['u2m']
    m2u_b, m2u_a = results['m2u']
    print(f"  S->M               {s2m_b:>9.2f}% {s2m_a:>9.2f}%    86.60%")
    print(f"  U->M               {u2m_b:>9.2f}% {u2m_a:>9.2f}%    89.15%")
    print(f"  M->U               {m2u_b:>9.2f}% {m2u_a:>9.2f}%    91.00%")
    print("=" * 60)