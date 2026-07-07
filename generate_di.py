import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from models.lenet import LeNet5
from config import config
from similarity_matrix import compute_similarity_matrix


def sample_dirichlet_vectors(alpha_k, beta, n_samples):
    scaled_alpha = (beta * alpha_k.cpu().numpy()).astype(np.float64)
    scaled_alpha = np.clip(scaled_alpha, 1e-6, None)
    samples = np.random.dirichlet(scaled_alpha, size=n_samples)
    return torch.FloatTensor(samples)


def generate_di_batch(model, target_softmax_batch, device):
    B = target_softmax_batch.shape[0]

    di_batch = torch.randn(
        B, config.IN_CHANNELS,
        config.IMAGE_SIZE, config.IMAGE_SIZE,
        device=device
    )
    di_batch.requires_grad_(True)

    target    = target_softmax_batch.to(device)
    optimizer = torch.optim.Adam([di_batch], lr=config.DI_LR)

    for iteration in range(config.DI_ITERATIONS):
        optimizer.zero_grad()

        logits       = model(di_batch, temperature=config.TEMPERATURE)
        pred_softmax = F.softmax(logits, dim=1)

        loss = -torch.mean(
            torch.sum(target * torch.log(pred_softmax + 1e-8), dim=1)
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            di_batch.clamp_(-2.5, 2.5)

    return di_batch.detach()

def generate_data_impressions(model, sim_matrix):
    print("\n" + "="*45)
    print("  STEP 3 — GENERATE DATA IMPRESSIONS")
    print("="*45)
    print(f"  Classes      : {config.NUM_CLASSES}")
    print(f"  DI per Class : {config.NUM_DI_PER_CLASS}")
    print(f"  Total DIs    : {config.NUM_DI_PER_CLASS * config.NUM_CLASSES}")
    print(f"  Beta Values  : {config.BETA_VALUES}")
    print(f"  Temperature  : {config.TEMPERATURE}")
    print(f"  Iterations   : {config.DI_ITERATIONS}")
    print(f"  Batch Size   : {config.DI_BATCH_SIZE}")
    print("="*45)

    model.eval()
~ 7
    all_di_images = []
    all_di_labels = []

    n_per_beta = config.NUM_DI_PER_CLASS // len(config.BETA_VALUES)

    for class_idx in range(config.NUM_CLASSES):
        alpha_k   = sim_matrix[class_idx]
        class_dis = []

        for beta in config.BETA_VALUES:
            sampled_vectors = sample_dirichlet_vectors(
                alpha_k, beta, n_per_beta
            )

            pbar = tqdm(
                range(0, n_per_beta, config.DI_BATCH_SIZE),
                desc=f"  Class {class_idx} | β={beta}"
            )

            for start_idx in pbar:
                end_idx       = min(start_idx + config.DI_BATCH_SIZE, n_per_beta)
                batch_targets = sampled_vectors[start_idx:end_idx]

                di_batch = generate_di_batch(
                    model, batch_targets, config.DEVICE
                )

                class_dis.append(di_batch.cpu())

                pbar.set_postfix({
                    'Done': f'{end_idx}/{n_per_beta}'
                })

        class_di_tensor = torch.cat(class_dis, dim=0)
        all_di_images.append(class_di_tensor)
        all_di_labels.extend([class_idx] * config.NUM_DI_PER_CLASS)

        print(f"  ✅ Class {class_idx}: {class_di_tensor.shape[0]} DIs done")

    all_di_images = torch.cat(all_di_images, dim=0)
    all_di_labels = torch.tensor(all_di_labels, dtype=torch.long)

    print(f"\n  Total DIs Shape : {all_di_images.shape}")
    print(f"  Labels Shape    : {all_di_labels.shape}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    save_path = f'{config.CHECKPOINT_DIR}data_impressions.pth'
    torch.save({
        'di_images' : all_di_images,
        'di_labels' : all_di_labels,
        'sim_matrix': sim_matrix,
    }, save_path)

    print(f"  ✅ Saved: {save_path}")
    return all_di_images, all_di_labels


def visualize_dis(di_images, di_labels, n_per_class=2):
    fig, axes = plt.subplots(
        n_per_class, config.NUM_CLASSES,
        figsize=(20, 4)
    )

    for class_idx in range(config.NUM_CLASSES):
        mask      = (di_labels == class_idx)
        class_dis = di_images[mask]

        for row in range(n_per_class):
            di = class_dis[row].squeeze().numpy()
            ax = axes[row][class_idx]
            ax.imshow(di, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(f'{class_idx}', fontsize=12)

    plt.suptitle(f'Data Impressions — {config.DATASET}', fontsize=14)
    plt.tight_layout()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = f'{config.RESULTS_DIR}data_impressions_viz.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ DI Visualization saved: {path}")
    plt.close()


if __name__ == "__main__":
    model = LeNet5().to(config.DEVICE)
    model.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    model.eval()

    sim_matrix = compute_similarity_matrix(model)
    dis, labels = generate_data_impressions(model, sim_matrix)
    visualize_dis(dis, labels)