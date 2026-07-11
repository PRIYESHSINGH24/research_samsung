"""
TABLE 4 — Adversarial Robustness (Section 4.1.7)
Paper: ALL 3 datasets, 4 models each
  Non-robust teacher, Student from non-robust DIs
  Robust teacher (PGD training), Student from robust DIs
  Attacks: FGSM, iFGSM, PGD
  Report: Anat, Aadv, F.R.
  MNIST/FMNIST: LeNet-5 teacher
  CIFAR: AlexNet teacher
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np, os
from models.lenet import LeNet5, LeNet5Half
from models.alexnet import AlexNet, AlexNetHalf
from config import config


# ═══════════════════════════════════
# ATTACKS
# ═══════════════════════════════════

def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad_(True)
    out = model(images); loss = nn.CrossEntropyLoss()(out, labels)
    model.zero_grad(); loss.backward()
    adv = images + epsilon * images.grad.sign()
    return adv.detach()

def ifgsm_attack(model, images, labels, epsilon, alpha=None, steps=10):
    if alpha is None: alpha = epsilon / steps
    adv = images.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv); loss = nn.CrossEntropyLoss()(out, labels)
        model.zero_grad(); loss.backward()
        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = (images + delta).detach()
    return adv

def pgd_attack(model, images, labels, epsilon, alpha=None, steps=20):
    if alpha is None: alpha = epsilon / 4
    adv = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv = adv.detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        out = model(adv); loss = nn.CrossEntropyLoss()(out, labels)
        model.zero_grad(); loss.backward()
        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - images, -epsilon, epsilon)
        adv = (images + delta).detach()
    return adv


def eval_attack(model, loader, attack_fn, epsilon):
    model.eval()
    correct_nat = correct_adv = fooled = total = 0
    for img, lab in loader:
        img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
        with torch.no_grad():
            nat_out = model(img); _, nat_pred = nat_out.max(1)
        adv_img = attack_fn(model, img, lab, epsilon)
        with torch.no_grad():
            adv_out = model(adv_img); _, adv_pred = adv_out.max(1)
        total += lab.size(0)
        correct_nat += nat_pred.eq(lab).sum().item()
        correct_adv += adv_pred.eq(lab).sum().item()
        fooled += (nat_pred != adv_pred).sum().item()
    anat = 100.*correct_nat/total
    aadv = 100.*correct_adv/total
    fr = 100.*fooled/total
    return anat, aadv, fr


# ═══════════════════════════════════
# PGD ADVERSARIAL TRAINING
# ═══════════════════════════════════

def pgd_train(model, train_loader, test_loader, epochs, lr, epsilon, name):
    print(f"\n  --- PGD Adversarial Training: {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for ep in range(1, epochs+1):
        model.train()
        for img, lab in tqdm(train_loader, leave=False, desc=f"Rob {ep}"):
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            adv_img = pgd_attack(model, img, lab, epsilon, steps=7)
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(adv_img), lab)
            loss.backward(); optimizer.step()
        acc = evaluate(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'{config.CHECKPOINT_DIR}{name}_robust.pth')
        if ep % 10 == 0: print(f"  Ep {ep}: {acc:.2f}% | Best: {best_acc:.2f}%")
    print(f"  Robust {name}: {best_acc:.2f}%")
    return best_acc


def evaluate(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
            _, p = model(img).max(1); t += lab.size(0); c += p.eq(lab).sum().item()
    return 100.*c/t


# ═══════════════════════════════════
# DI GENERATION + ZSKD (reuse from tables 1-3)
# ═══════════════════════════════════

def compute_sim(model):
    w = model.get_final_weights()
    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    row_min = s.min(dim=1, keepdim=True).values
    row_max = s.max(dim=1, keepdim=True).values
    s = (s - row_min) / (row_max - row_min + 1e-8)
    return s.cpu().numpy()

def gen_di_batch(model, targets, dev, ch=1):
    B = targets.shape[0]; di = torch.randn(B,ch,32,32,device=dev); di.requires_grad_(True)
    targets = targets.to(dev); opt = torch.optim.Adam([di], lr=0.1)
    for _ in range(1500):
        opt.zero_grad(); logits = model(di, temperature=20); pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward(); opt.step()
        with torch.no_grad(): di.clamp_(-0.4242, 2.8215)
    return di.detach()

def generate_dis(model, n_per_class, ch=1):
    sim = compute_sim(model); all_dis, all_lab = [], []
    for c in range(10):
        cdis = []
        for beta in [1.0, 0.1]:
            n = n_per_class // 2; a = np.clip(sim[c]*beta, 1e-3, None)
            tgts = torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)
            for s in tqdm(range(0,n,32), desc=f"  C{c} β={beta}"):
                e = min(s+32,n); cdis.append(gen_di_batch(model, tgts[s:e], config.DEVICE, ch).cpu())
        ct = torch.cat(cdis,0); all_dis.append(ct); all_lab.extend([c]*n_per_class)
        print(f"  ✅ C{c}: {ct.shape[0]}")
    return torch.cat(all_dis,0), torch.tensor(all_lab, dtype=torch.long)

def train_zskd(teacher, student, dis, labels, test_loader, epochs=500):
    dl = DataLoader(torch.utils.data.TensorDataset(dis, labels), batch_size=256, shuffle=True, num_workers=0)
    opt = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        student.train()
        for b, _ in dl:
            b = b.to(config.DEVICE)
            with torch.no_grad(): tl = teacher(b)
            sl = student(b)
            loss = F.kl_div(F.log_softmax(sl/20,1), F.softmax(tl/20,1), reduction='batchmean')*400
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); acc = evaluate(student, test_loader)
        if acc > best: best = acc
        if ep % 100 == 0: print(f"    ZSKD Ep {ep}: {acc:.2f}% | Best: {best:.2f}%")
    return best


# ═══════════════════════════════════
# MAIN
# ═══════════════════════════════════

if __name__ == "__main__":
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print("="*70)
    print("  TABLE 4 — ADVERSARIAL ROBUSTNESS (ALL 3 DATASETS)")
    print("="*70)

    results = {}

    # ── MNIST ──
    print("\n" + "="*50 + "\n  MNIST\n" + "="*50)
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    train_data = datasets.MNIST('data/', train=True, download=True, transform=tf)
    test_data = datasets.MNIST('data/', train=False, transform=tf)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    eps_mnist = 0.3

    # Non-robust teacher (load from Table 1)
    nr_teacher = LeNet5().to(config.DEVICE)
    if os.path.exists(f'{config.CHECKPOINT_DIR}mnist_teacher.pth'):
        nr_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}mnist_teacher.pth', map_location=config.DEVICE))
        print("  Loaded non-robust MNIST teacher")
    else:
        print("  Training non-robust MNIST teacher...")
        opt = optim.Adam(nr_teacher.parameters(), lr=0.001)
        for ep in range(1, 101):
            nr_teacher.train()
            for img, lab in tqdm(train_loader, leave=False):
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt.zero_grad(); nn.CrossEntropyLoss()(nr_teacher(img), lab).backward(); opt.step()
        torch.save(nr_teacher.state_dict(), f'{config.CHECKPOINT_DIR}mnist_teacher.pth')
    nr_teacher.eval()

    # Non-robust teacher attacks
    print("  Non-robust teacher attacks...")
    anat, aadv_f, fr_f = eval_attack(nr_teacher, test_loader, fgsm_attack, eps_mnist)
    _, aadv_i, fr_i = eval_attack(nr_teacher, test_loader, ifgsm_attack, eps_mnist)
    _, aadv_p, fr_p = eval_attack(nr_teacher, test_loader, pgd_attack, eps_mnist)
    results['mnist_nr_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]
    print(f"  NR Teacher: Anat={anat:.2f} FGSM={aadv_f:.2f} iFGSM={aadv_i:.2f} PGD={aadv_p:.2f}")

    # Student from non-robust DIs
    print("  Generating DIs from non-robust teacher...")
    dis, labs = generate_dis(nr_teacher, 2400, ch=1)
    nr_student = LeNet5Half().to(config.DEVICE)
    zskd_acc = train_zskd(nr_teacher, nr_student, dis, labs, test_loader, epochs=300)
    anat, aadv_f, fr_f = eval_attack(nr_student, test_loader, fgsm_attack, eps_mnist)
    _, aadv_i, fr_i = eval_attack(nr_student, test_loader, ifgsm_attack, eps_mnist)
    _, aadv_p, fr_p = eval_attack(nr_student, test_loader, pgd_attack, eps_mnist)
    results['mnist_nr_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]
    print(f"  NR Student: Anat={anat:.2f} FGSM={aadv_f:.2f} iFGSM={aadv_i:.2f} PGD={aadv_p:.2f}")

    # Robust teacher
    rob_teacher = LeNet5().to(config.DEVICE)
    pgd_train(rob_teacher, train_loader, test_loader, epochs=50, lr=0.001, epsilon=eps_mnist, name='mnist')
    rob_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}mnist_robust.pth', map_location=config.DEVICE))
    rob_teacher.eval()
    anat, aadv_f, fr_f = eval_attack(rob_teacher, test_loader, fgsm_attack, eps_mnist)
    _, aadv_i, fr_i = eval_attack(rob_teacher, test_loader, ifgsm_attack, eps_mnist)
    _, aadv_p, fr_p = eval_attack(rob_teacher, test_loader, pgd_attack, eps_mnist)
    results['mnist_r_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]
    print(f"  Rob Teacher: Anat={anat:.2f} FGSM={aadv_f:.2f} iFGSM={aadv_i:.2f} PGD={aadv_p:.2f}")

    # Student from robust DIs
    print("  Generating DIs from robust teacher...")
    dis_r, labs_r = generate_dis(rob_teacher, 2400, ch=1)
    rob_student = LeNet5Half().to(config.DEVICE)
    train_zskd(rob_teacher, rob_student, dis_r, labs_r, test_loader, epochs=300)
    anat, aadv_f, fr_f = eval_attack(rob_student, test_loader, fgsm_attack, eps_mnist)
    _, aadv_i, fr_i = eval_attack(rob_student, test_loader, ifgsm_attack, eps_mnist)
    _, aadv_p, fr_p = eval_attack(rob_student, test_loader, pgd_attack, eps_mnist)
    results['mnist_r_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]
    print(f"  Rob Student: Anat={anat:.2f} FGSM={aadv_f:.2f} iFGSM={aadv_i:.2f} PGD={aadv_p:.2f}")

    # ── FMNIST ──
    print("\n" + "="*50 + "\n  FASHION-MNIST\n" + "="*50)
    tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.2860,),(0.3530,))])
    train_data = datasets.FashionMNIST('data/', train=True, download=True, transform=tf)
    test_data = datasets.FashionMNIST('data/', train=False, transform=tf)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    eps_fmnist = 0.1

    nr_teacher = LeNet5().to(config.DEVICE)
    if os.path.exists(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth'):
        nr_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth', map_location=config.DEVICE))
    else:
        opt = optim.Adam(nr_teacher.parameters(), lr=0.001)
        for ep in range(1, 101):
            nr_teacher.train()
            for img, lab in tqdm(train_loader, leave=False):
                img, lab = img.to(config.DEVICE), lab.to(config.DEVICE)
                opt.zero_grad(); nn.CrossEntropyLoss()(nr_teacher(img), lab).backward(); opt.step()
        torch.save(nr_teacher.state_dict(), f'{config.CHECKPOINT_DIR}fmnist_teacher.pth')
    nr_teacher.eval()

    anat, aadv_f, fr_f = eval_attack(nr_teacher, test_loader, fgsm_attack, eps_fmnist)
    _, aadv_i, fr_i = eval_attack(nr_teacher, test_loader, ifgsm_attack, eps_fmnist)
    _, aadv_p, fr_p = eval_attack(nr_teacher, test_loader, pgd_attack, eps_fmnist)
    results['fmnist_nr_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]
    print(f"  NR Teacher: Anat={anat:.2f} FGSM={aadv_f:.2f} iFGSM={aadv_i:.2f} PGD={aadv_p:.2f}")

    dis, labs = generate_dis(nr_teacher, 4800, ch=1)
    nr_student = LeNet5Half().to(config.DEVICE)
    train_zskd(nr_teacher, nr_student, dis, labs, test_loader, 300)
    anat, aadv_f, fr_f = eval_attack(nr_student, test_loader, fgsm_attack, eps_fmnist)
    _, aadv_i, fr_i = eval_attack(nr_student, test_loader, ifgsm_attack, eps_fmnist)
    _, aadv_p, fr_p = eval_attack(nr_student, test_loader, pgd_attack, eps_fmnist)
    results['fmnist_nr_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    rob_teacher = LeNet5().to(config.DEVICE)
    pgd_train(rob_teacher, train_loader, test_loader, 50, 0.001, eps_fmnist, 'fmnist')
    rob_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}fmnist_robust.pth', map_location=config.DEVICE)); rob_teacher.eval()
    anat, aadv_f, fr_f = eval_attack(rob_teacher, test_loader, fgsm_attack, eps_fmnist)
    _, aadv_i, fr_i = eval_attack(rob_teacher, test_loader, ifgsm_attack, eps_fmnist)
    _, aadv_p, fr_p = eval_attack(rob_teacher, test_loader, pgd_attack, eps_fmnist)
    results['fmnist_r_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    dis_r, labs_r = generate_dis(rob_teacher, 4800, ch=1)
    rob_student = LeNet5Half().to(config.DEVICE)
    train_zskd(rob_teacher, rob_student, dis_r, labs_r, test_loader, 300)
    anat, aadv_f, fr_f = eval_attack(rob_student, test_loader, fgsm_attack, eps_fmnist)
    _, aadv_i, fr_i = eval_attack(rob_student, test_loader, ifgsm_attack, eps_fmnist)
    _, aadv_p, fr_p = eval_attack(rob_student, test_loader, pgd_attack, eps_fmnist)
    results['fmnist_r_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    # ── CIFAR ──
    print("\n" + "="*50 + "\n  CIFAR-10\n" + "="*50)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    train_data = datasets.CIFAR10('data/', train=True, download=False, transform=tf)
    test_data = datasets.CIFAR10('data/', train=False, transform=tf)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    eps_cifar = 8/255

    nr_teacher = AlexNet().to(config.DEVICE)
    if os.path.exists(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth'):
        nr_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
    nr_teacher.eval()

    anat, aadv_f, fr_f = eval_attack(nr_teacher, test_loader, fgsm_attack, eps_cifar)
    _, aadv_i, fr_i = eval_attack(nr_teacher, test_loader, ifgsm_attack, eps_cifar)
    _, aadv_p, fr_p = eval_attack(nr_teacher, test_loader, pgd_attack, eps_cifar)
    results['cifar_nr_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    dis, labs = generate_dis(nr_teacher, 4000, ch=3)
    nr_student = AlexNetHalf().to(config.DEVICE)
    train_zskd(nr_teacher, nr_student, dis, labs, test_loader, 300)
    anat, aadv_f, fr_f = eval_attack(nr_student, test_loader, fgsm_attack, eps_cifar)
    _, aadv_i, fr_i = eval_attack(nr_student, test_loader, ifgsm_attack, eps_cifar)
    _, aadv_p, fr_p = eval_attack(nr_student, test_loader, pgd_attack, eps_cifar)
    results['cifar_nr_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    rob_teacher = AlexNet().to(config.DEVICE)
    pgd_train(rob_teacher, train_loader, test_loader, 50, 0.001, eps_cifar, 'cifar')
    rob_teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_robust.pth', map_location=config.DEVICE)); rob_teacher.eval()
    anat, aadv_f, fr_f = eval_attack(rob_teacher, test_loader, fgsm_attack, eps_cifar)
    _, aadv_i, fr_i = eval_attack(rob_teacher, test_loader, ifgsm_attack, eps_cifar)
    _, aadv_p, fr_p = eval_attack(rob_teacher, test_loader, pgd_attack, eps_cifar)
    results['cifar_r_t'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    dis_r, labs_r = generate_dis(rob_teacher, 4000, ch=3)
    rob_student = AlexNetHalf().to(config.DEVICE)
    train_zskd(rob_teacher, rob_student, dis_r, labs_r, test_loader, 300)
    anat, aadv_f, fr_f = eval_attack(rob_student, test_loader, fgsm_attack, eps_cifar)
    _, aadv_i, fr_i = eval_attack(rob_student, test_loader, ifgsm_attack, eps_cifar)
    _, aadv_p, fr_p = eval_attack(rob_student, test_loader, pgd_attack, eps_cifar)
    results['cifar_r_s'] = [anat, aadv_f, fr_f, aadv_i, fr_i, aadv_p, fr_p]

    # ── PRINT TABLE ──
    print(f"\n{'='*80}")
    print(f"  TABLE 4 — ADVERSARIAL ROBUSTNESS RESULTS")
    print(f"{'='*80}")
    print(f"  {'Dataset':<10s} {'Model':<35s} {'Anat':>6s} {'FGSM':>6s} {'FR':>6s} {'iFGSM':>6s} {'FR':>6s} {'PGD':>6s} {'FR':>6s}")
    print(f"  {'-'*75}")
    for ds in ['mnist', 'fmnist', 'cifar']:
        for key, label in [(f'{ds}_nr_t', 'Non-robust teacher'),
                           (f'{ds}_nr_s', 'Student from NR DIs'),
                           (f'{ds}_r_t', 'Robust teacher'),
                           (f'{ds}_r_s', 'Student from Rob DIs')]:
            r = results[key]
            print(f"  {ds.upper():<10s} {label:<35s} {r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.2f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f}")
        print(f"  {'-'*75}")
    print(f"{'='*80}")