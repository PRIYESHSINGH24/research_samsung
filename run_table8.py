"""
TABLE 8 / FIGURE 8 — Continual Learning (Section 4.3)
Paper Settings:
  - CIFAR-100, ResNet-32
  - Step size: 20 classes (5 steps: 20→40→60→80→100)
  - 2400 DIs TOTAL (not per class!)
  - Normalize: mean=0.5, std=0.5
  - SGD, momentum=0.9, weight_decay=0.0005
  - New class model lr=0.1
  - Combined model lr=0.01 (last step=0.001)
  - LR reduced by 1/5 every 70 epochs
  - Max 500 epochs
  - Dual distillation loss from DMC [23]
  - Initialize combined with old class weights
  - Mean of 5 trials
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np, os, copy
import matplotlib.pyplot as plt
from config import config


# ═══════════════════════════════════
# ResNet-32 for CIFAR-100 (Paper: same as [19][20][22][23])
# ═══════════════════════════════════

class BasicBlock32(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet32(nn.Module):
    """ResNet-32 for CIFAR-100"""
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 5, 1)
        self.layer2 = self._make_layer(16, 32, 5, 2)
        self.layer3 = self._make_layer(32, 64, 5, 2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock32(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock32(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.avg_pool2d(x, x.size(3))
        x = x.view(x.size(0), -1)
        return self.fc(x) / temperature

    def get_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.avg_pool2d(x, x.size(3))
        return x.view(x.size(0), -1)

    def get_final_weights(self):
        return self.fc.weight.data.clone()


# ═══════════════════════════════════
# DI GENERATION
# ═══════════════════════════════════

def compute_sim(model, num_classes):
    w = model.get_final_weights()[:num_classes]
    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    return ((s-s.min())/(s.max()-s.min()+1e-8)).cpu().numpy()

def gen_di_batch(model, targets, dev, num_classes):
    B = targets.shape[0]; di = torch.randn(B,3,32,32,device=dev); di.requires_grad_(True)
    targets = targets.to(dev); opt = torch.optim.Adam([di], lr=0.01)
    for _ in range(1500):
        opt.zero_grad()
        logits = model(di, temperature=20)[:, :num_classes]
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward(); opt.step()
        with torch.no_grad(): di.clamp_(-1, 1)
    return di.detach()

def generate_dis_cl(model, num_classes, total_dis=2400):
    """Paper: 2400 DIs TOTAL, distributed across old classes"""
    n_per_class = total_dis // num_classes
    sim = compute_sim(model, num_classes)
    all_dis, all_lab = [], []
    for c in range(num_classes):
        cdis = []
        for beta in [1.0, 0.1]:
            n = max(n_per_class // 2, 1)
            a = np.clip(sim[c]*beta, 1e-3, None)
            tgts = torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)
            for s in range(0, n, 32):
                e = min(s+32, n)
                cdis.append(gen_di_batch(model, tgts[s:e], config.DEVICE, num_classes).cpu())
        ct = torch.cat(cdis, 0); all_dis.append(ct); all_lab.extend([c]*ct.shape[0])
    return torch.cat(all_dis,0), torch.tensor(all_lab, dtype=torch.long)


# ═══════════════════════════════════
# DATA
# ═══════════════════════════════════

def get_cifar100_loaders(classes):
    """Paper: normalized with channel mean=0.5, std=0.5"""
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    tf_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train = datasets.CIFAR100('data/', train=True, download=True, transform=tf)
    test = datasets.CIFAR100('data/', train=False, download=True, transform=tf_test)

    train_idx = [i for i, (_, y) in enumerate(train) if y in classes]
    test_idx = [i for i, (_, y) in enumerate(test) if y in classes]

    train_loader = DataLoader(Subset(train, train_idx), batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(Subset(test, test_idx), batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader

def get_all_test_loader(classes):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    test = datasets.CIFAR100('data/', train=False, download=True, transform=tf)
    idx = [i for i, (_, y) in enumerate(test) if y in classes]
    return DataLoader(Subset(test, idx), batch_size=256, shuffle=False, num_workers=0)


def evaluate_cl(model, loader, all_classes):
    model.eval(); c=t=0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            out = model(img)[:, all_classes]
            _, pred = out.max(1)
            # Map prediction index back to class
            pred_class = torch.tensor([all_classes[p] for p in pred.cpu()], device=config.DEVICE)
            t += lab.size(0); c += pred_class.eq(lab).sum().item()
    return 100.*c/t


# ═══════════════════════════════════
# CONTINUAL LEARNING PIPELINE
# ═══════════════════════════════════

def run_one_trial(trial_id):
    print(f"\n  ══ TRIAL {trial_id} ══")
    step_size = 20
    steps = 5
    step_accs = []

    # Step 1: Train on first 20 classes
    classes_so_far = list(range(0, step_size))
    print(f"\n  Step 1: Classes {classes_so_far[0]}-{classes_so_far[-1]}")

    train_loader, _ = get_cifar100_loaders(classes_so_far)
    model = ResNet32(num_classes=100).to(config.DEVICE)
    # Paper: new class lr=0.1
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

    for ep in range(1, 201):
        model.train()
        for img, lab in train_loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            opt.zero_grad(); nn.CrossEntropyLoss()(model(img), lab).backward(); opt.step()
        # LR decay: 1/5 every 70 epochs
        if ep % 70 == 0:
            for g in opt.param_groups: g['lr'] /= 5

    test_loader = get_all_test_loader(classes_so_far)
    acc = evaluate_cl(model, test_loader, classes_so_far)
    step_accs.append(acc)
    print(f"  Step 1 Acc: {acc:.2f}%")
    old_model = copy.deepcopy(model)

    # Steps 2-5
    for step in range(2, steps + 1):
        new_classes = list(range((step-1)*step_size, step*step_size))
        classes_so_far = list(range(0, step*step_size))
        print(f"\n  Step {step}: Adding classes {new_classes[0]}-{new_classes[-1]}")

        # Generate DIs from old model
        print(f"  Generating 2400 DIs for {len(classes_so_far)-step_size} old classes...")
        old_classes = list(range(0, (step-1)*step_size))
        dis, di_labs = generate_dis_cl(old_model, len(old_classes), total_dis=2400)

        # Train new class model
        new_train, _ = get_cifar100_loaders(new_classes)
        new_model = ResNet32(num_classes=100).to(config.DEVICE)
        opt_new = optim.SGD(new_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        for ep in range(1, 201):
            new_model.train()
            for img, lab in new_train:
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt_new.zero_grad(); nn.CrossEntropyLoss()(new_model(img), lab).backward(); opt_new.step()
            if ep % 70 == 0:
                for g in opt_new.param_groups: g['lr'] /= 5

        # Combined model with dual distillation
        combined = ResNet32(num_classes=100).to(config.DEVICE)
        combined.load_state_dict(old_model.state_dict())  # Initialize with old weights

        # Paper: lr=0.01, last step lr=0.001
        lr_combined = 0.001 if step == steps else 0.01
        opt_c = optim.SGD(combined.parameters(), lr=lr_combined, momentum=0.9, weight_decay=0.0005)

        # Create combined dataset: DIs + new class data
        aug_tf = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32,4),
                                     transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        new_train_data, _ = get_cifar100_loaders(new_classes)

        old_model.eval(); new_model.eval()

        for ep in range(1, 501):
            combined.train()

            # Train on DIs (old class distillation)
            di_perm = torch.randperm(len(dis))
            for s in range(0, len(dis), 128):
                e = min(s+128, len(dis))
                idx = di_perm[s:e]
                batch = dis[idx].to(config.DEVICE)
                with torch.no_grad():
                    old_logits = old_model(batch)
                    new_logits = new_model(batch)
                    target_logits = torch.cat([old_logits[:, old_classes],
                                               new_logits[:, new_classes]], dim=1)
                comb_logits = combined(batch)
                comb_selected = torch.cat([comb_logits[:, old_classes],
                                           comb_logits[:, new_classes]], dim=1)
                # Dual distillation: MSE loss
                loss = F.mse_loss(comb_selected, target_logits)
                opt_c.zero_grad(); loss.backward(); opt_c.step()

            # Train on new class data
            for img, lab in new_train_data:
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                with torch.no_grad():
                    old_logits = old_model(img)
                    new_logits = new_model(img)
                    target_logits = torch.cat([old_logits[:, old_classes],
                                               new_logits[:, new_classes]], dim=1)
                comb_logits = combined(img)
                comb_selected = torch.cat([comb_logits[:, old_classes],
                                           comb_logits[:, new_classes]], dim=1)
                loss = F.mse_loss(comb_selected, target_logits)
                opt_c.zero_grad(); loss.backward(); opt_c.step()

            # LR decay
            if ep % 70 == 0:
                for g in opt_c.param_groups: g['lr'] /= 5

            if ep % 100 == 0:
                test_loader = get_all_test_loader(classes_so_far)
                acc = evaluate_cl(combined, test_loader, classes_so_far)
                print(f"    Combined Ep {ep}: {acc:.2f}%")

        test_loader = get_all_test_loader(classes_so_far)
        acc = evaluate_cl(combined, test_loader, classes_so_far)
        step_accs.append(acc)
        print(f"  Step {step} Final: {acc:.2f}%")
        old_model = copy.deepcopy(combined)

    return step_accs


if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("="*60)
    print("  TABLE 8 / FIGURE 8 — CONTINUAL LEARNING")
    print("  CIFAR-100, ResNet-32, step=20, 5 trials")
    print("="*60)

    # Paper: mean of 5 trials
    all_trials = []
    for trial in range(1, 6):
        accs = run_one_trial(trial)
        all_trials.append(accs)
        print(f"\n  Trial {trial} results: {[f'{a:.2f}' for a in accs]}")

    # Compute mean
    mean_accs = np.mean(all_trials, axis=0)
    std_accs = np.std(all_trials, axis=0)

    print(f"\n{'='*60}")
    print(f"  CONTINUAL LEARNING RESULTS (Mean of 5 trials)")
    print(f"{'='*60}")
    classes_list = [20, 40, 60, 80, 100]
    for i, (n, m, s) in enumerate(zip(classes_list, mean_accs, std_accs)):
        print(f"  {n} classes: {m:.2f}% ± {s:.2f}%")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(classes_list, mean_accs, 'bo-', label='Ours (Data Impressions)', linewidth=2)
    plt.xlabel('Number of Classes'); plt.ylabel('Classification Accuracy (%)')
    plt.title('Continual Learning on CIFAR-100')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{config.RESULTS_DIR}figure8_continual_learning.png', dpi=150)
    print(f"  Saved: figure8_continual_learning.png")