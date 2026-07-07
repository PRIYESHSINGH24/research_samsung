import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from models.vgg import VGG19
from config import config
from similarity_matrix import compute_similarity_matrix
from generate_di import sample_dirichlet_vectors


def generate_di_batch_paper(model, target_softmax_batch, device, iterations=1500):
    """
    Paper exact settings:
    - LR = 10 with linear decay
    - 1500 iterations
    """
    B = target_softmax_batch.shape[0]

    di_batch = torch.randn(
        B, 3, 32, 32,
        device=device
    )
    di_batch.requires_grad_(True)

    target = target_softmax_batch.to(device)
    
    # Paper: lr=10 with linear decay
    optimizer = torch.optim.Adam([di_batch], lr=10.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.001,
        total_iters=iterations
    )

    for iteration in range(iterations):
        optimizer.zero_grad()
        logits = model(di_batch, temperature=20)
        pred_softmax = F.softmax(logits, dim=1)
        loss = -torch.mean(
            torch.sum(target * torch.log(pred_softmax + 1e-8), dim=1)
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            di_batch.clamp_(-3.0, 3.0)

    return di_batch.detach()


def regenerate_cifar_dis():
    print("="*50)
    print("  REGENERATING CIFAR DIs")
    print("  Paper exact: LR=10, Linear decay")
    print("="*50)

    # Load teacher
    teacher = VGG19().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}VGG19_cifar_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    # Similarity matrix
    sim_matrix = compute_similarity_matrix(teacher)

    num_per_class = 4000
    beta_values = [1.0, 0.1]
    n_per_beta = num_per_class // len(beta_values)
    di_batch_size = 32

    all_images = []
    all_labels = []

    print(f"\n  Generating {num_per_class * 10} DIs total...")
    print(f"  Per class: {num_per_class}")
    print(f"  Beta values: {beta_values}")

    for class_idx in range(10):
        print(f"\n  ===== Class {class_idx}/9 =====")
        alpha_k = sim_matrix[class_idx]
        class_dis = []

        for beta in beta_values:
            sampled = sample_dirichlet_vectors(alpha_k, beta, n_per_beta)

            pbar = tqdm(
                range(0, n_per_beta, di_batch_size),
                desc=f"  β={beta}"
            )
            for start in pbar:
                end = min(start + di_batch_size, n_per_beta)
                batch_targets = sampled[start:end]
                di_batch = generate_di_batch_paper(
                    teacher, batch_targets, config.DEVICE
                )
                class_dis.append(di_batch.cpu())

        class_tensor = torch.cat(class_dis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ Class {class_idx} done: {class_tensor.shape[0]} DIs")
        
        # Quick quality check
        with torch.no_grad():
            sample = class_tensor[:100].to(config.DEVICE)
            logits = teacher(sample)
            preds = logits.argmax(1)
            class_acc = (preds == class_idx).float().mean().item() * 100
            print(f"  Quality check: {class_acc:.1f}% predicted as class {class_idx}")

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    print(f"\n  Total DIs: {all_images.shape}")

    # Save with new name
    save_path = f'{config.CHECKPOINT_DIR}cifar_dis_v2.pth'
    torch.save({
        'di_images': all_images,
        'di_labels': all_labels,
    }, save_path)
    print(f"  ✅ Saved to: {save_path}")

    # Final quality check
    print(f"\n  Final Quality Check on All DIs...")
    correct = 0
    total = 0
    confidences = []
    
    with torch.no_grad():
        for i in range(0, len(all_images), 256):
            batch = all_images[i:i+256].to(config.DEVICE)
            labels = all_labels[i:i+256].to(config.DEVICE)
            logits = teacher(batch)
            probs = F.softmax(logits, dim=1)
            max_probs, predicted = probs.max(1)
            confidences.extend(max_probs.cpu().numpy().tolist())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    teacher_acc_on_dis = 100. * correct / total
    avg_confidence = np.mean(confidences) * 100
    
    print(f"\n  Teacher Acc on New DIs : {teacher_acc_on_dis:.2f}%")
    print(f"  Previous (Old DIs)     : 37.46%")
    print(f"  Avg Confidence         : {avg_confidence:.2f}%")


if __name__ == "__main__":
    regenerate_cifar_dis()