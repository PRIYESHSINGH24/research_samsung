"""
validate_di.py  —  FAST DI-generation sanity check for Table 5 (VGG-19 / CIFAR-10)

Purpose: before spending ~24 hours regenerating 40,000 DIs, generate a SMALL
batch with the FIXED generation code and confirm the teacher actually
recognizes them and the predictions are class-balanced.

Two fixes over the old run_table5 generation:
  1. Class-similarity matrix is min-max normalized PER ROW (paper Sec 3.1),
     not globally -> each row c_k is a proper Dirichlet concentration alpha_k.
  2. DIs are clamped to the VALID per-channel normalized range that real
     [0,1] pixels occupy after CIFAR normalization, instead of the arbitrary
     [-3, 3]. This keeps DIs on the real-data manifold (old DIs had std ~2.49
     vs real ~1.0 and were pinned at the clamp edges -> out of distribution).

GO / NO-GO:
  - Teacher acc on these DIs  > ~85%  AND  predictions spread across classes
      -> generation is healthy, safe to launch the full 24 h regen.
  - Still low / collapsed onto a few classes
      -> do NOT launch the 24 h run; we iterate on generation first.

Run (use your GPU env — on CPU this is very slow):
    python validate_di.py
Adjust N_PER_CLASS below for a faster/slower check.
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm

from models.vgg import VGG19
from config import config

# ---- knobs ----
N_PER_CLASS   = 50        # 50 * 10 = 500 DIs. Lower to 20 for a quick smoke test.
DI_ITERS      = 1500      # keep = full generation so the check is representative
DI_LR         = 10.0      # paper VGG setting
DI_BATCH      = 32
TEMPERATURE   = 20
BETAS         = [0.1, 1.0]
TEACHER_CKPT  = f'{config.CHECKPOINT_DIR}vgg19_cifar_teacher.pth'

# CIFAR-10 normalization used to train the teacher
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
# valid per-channel range that real [0,1] pixels map to after normalization
CLAMP_MIN = ((0.0 - CIFAR_MEAN) / CIFAR_STD)
CLAMP_MAX = ((1.0 - CIFAR_MEAN) / CIFAR_STD)


def compute_similarity_matrix(model):
    """FIX #1: per-row min-max normalization (paper Sec 3.1)."""
    w = model.get_final_weights()
    sim = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    row_min = sim.min(dim=1, keepdim=True).values
    row_max = sim.max(dim=1, keepdim=True).values
    sim = (sim - row_min) / (row_max - row_min + 1e-8)
    return sim.cpu().numpy()


def sample_dirichlet(alpha, beta, n):
    a = np.clip(alpha * beta, 1e-3, None)
    return torch.tensor(np.random.dirichlet(a, n), dtype=torch.float32)


def generate_di_batch(model, targets, dev, cmin, cmax):
    """FIX #2: clamp to valid per-channel range instead of [-3, 3]."""
    B = targets.shape[0]
    di = torch.randn(B, 3, 32, 32, device=dev)
    with torch.no_grad():
        di = torch.max(torch.min(di, cmax), cmin)   # start in-range
    di.requires_grad_(True)
    targets = targets.to(dev)

    optimizer = torch.optim.Adam([di], lr=DI_LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.001, total_iters=DI_ITERS)

    for _ in range(DI_ITERS):
        optimizer.zero_grad()
        logits = model(di, temperature=TEMPERATURE)
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            di.data = torch.max(torch.min(di.data, cmax), cmin)
    return di.detach()


def main():
    dev = config.DEVICE
    print("=" * 55)
    print("  DI GENERATION VALIDATION  (fixed generation, small batch)")
    print("=" * 55)
    print(f"  Device        : {dev}")
    print(f"  DIs per class : {N_PER_CLASS}  (total {N_PER_CLASS * 10})")
    print(f"  Iterations    : {DI_ITERS}   LR: {DI_LR} (linear decay)")
    if str(dev) == 'cpu':
        print("  ⚠️  Running on CPU — this will be slow. Use your GPU env if possible.")
    print("=" * 55)

    teacher = VGG19().to(dev)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=dev))
    teacher.eval()

    cmin, cmax = CLAMP_MIN.to(dev), CLAMP_MAX.to(dev)
    sim = compute_similarity_matrix(teacher)

    all_dis, all_labels = [], []
    for cls in range(10):
        alpha = sim[cls]
        class_dis = []
        for beta in BETAS:
            n = N_PER_CLASS // len(BETAS)
            targets = sample_dirichlet(alpha, beta, n)
            for start in tqdm(range(0, n, DI_BATCH), desc=f"  C{cls} β={beta}", leave=False):
                end = min(start + DI_BATCH, n)
                di = generate_di_batch(teacher, targets[start:end], dev, cmin, cmax)
                class_dis.append(di.cpu())
        ct = torch.cat(class_dis, dim=0)
        all_dis.append(ct)
        all_labels.extend([cls] * ct.shape[0])
        print(f"  ✅ Class {cls}: {ct.shape[0]} DIs")

    dis = torch.cat(all_dis, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)

    # ---- diagnostics ----
    print(f"\n  DI Range : [{dis.min():.2f}, {dis.max():.2f}]  (real ≈ [-2.4, 2.7])")
    print(f"  DI Std   : {dis.std():.3f}  (real ≈ 1.0)")

    correct, total = 0, 0
    confidences, pred_counts = [], Counter()
    with torch.no_grad():
        for i in range(0, len(dis), 256):
            batch = dis[i:i + 256].to(dev)
            lab = labels[i:i + 256].to(dev)
            probs = F.softmax(teacher(batch), dim=1)
            maxp, pred = probs.max(1)
            confidences.extend(maxp.cpu().tolist())
            pred_counts.update(pred.cpu().tolist())
            total += lab.size(0)
            correct += pred.eq(lab).sum().item()

    acc = 100.0 * correct / total
    conf = 100.0 * float(np.mean(confidences))

    print(f"\n  Teacher Acc on DIs : {acc:.2f}%")
    print(f"  Avg Confidence     : {conf:.2f}%")
    print(f"\n  Teacher prediction spread across classes:")
    for c in range(10):
        bar = "█" * int(pred_counts.get(c, 0) / max(1, total) * 50)
        print(f"    class {c}: {pred_counts.get(c, 0):4d}  {bar}")

    # class-balance metric: how many classes get a meaningful share
    shares = np.array([pred_counts.get(c, 0) for c in range(10)]) / total
    active = int((shares > 0.02).sum())

    print("\n" + "=" * 55)
    print("  VERDICT")
    print("=" * 55)
    if acc >= 85 and active >= 8:
        print("  ✅ HEALTHY — teacher recognizes DIs and predictions are")
        print("     spread across classes. Safe to launch the full 24 h regen.")
    elif acc >= 60:
        print("  ⚠️  IMPROVED but not there yet. Don't launch 24 h run.")
        print(f"     acc={acc:.1f}% (want ≥85), active classes={active}/10 (want ≥8).")
    else:
        print("  ❌ STILL BROKEN. Do NOT launch the 24 h run.")
        print(f"     acc={acc:.1f}%, active classes={active}/10 — iterate on generation.")
    print("=" * 55)


if __name__ == "__main__":
    main()
