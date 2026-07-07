import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from models.lenet import LeNet5
from config import config


def compute_similarity_matrix(model):
    print("\n" + "="*45)
    print("  STEP 2 — SIMILARITY MATRIX")
    print("="*45)

    model.eval()

    with torch.no_grad():
        weights = model.get_final_weights()
        print(f"  Weight Shape    : {weights.shape}")

        weights_norm = F.normalize(weights, p=2, dim=1)
        sim_matrix   = torch.mm(weights_norm, weights_norm.t())

        print(f"  Before Norm Min : {sim_matrix.min():.4f}")
        print(f"  Before Norm Max : {sim_matrix.max():.4f}")

        for i in range(sim_matrix.shape[0]):
            row     = sim_matrix[i]
            row_min = row.min()
            row_max = row.max()
            sim_matrix[i] = (row - row_min) / (row_max - row_min + 1e-8)

        print(f"  After Norm Min  : {sim_matrix.min():.4f}")
        print(f"  After Norm Max  : {sim_matrix.max():.4f}")

    print(f"  ✅ Similarity Matrix Ready!")
    return sim_matrix


def visualize_similarity_matrix(sim_matrix, save=True):
    labels = [str(i) for i in range(10)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        sim_matrix.cpu().numpy(),
        cmap='viridis',
        vmin=0, vmax=1
    )
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(10):
        for j in range(10):
            val = sim_matrix[i, j].item()
            ax.text(j, i, f'{val:.2f}',
                   ha='center', va='center',
                   fontsize=7,
                   color='white' if val < 0.5 else 'black')

    ax.set_title(f'Class Similarity Matrix — {config.DATASET}')
    plt.tight_layout()

    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        path = f'{config.RESULTS_DIR}similarity_matrix.png'
        plt.savefig(path, dpi=150)
        print(f"  ✅ Saved: {path}")

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

    sim = compute_similarity_matrix(model)
    visualize_similarity_matrix(sim)

    print("\nSimilarity Matrix Values:")
    print(sim.cpu().numpy().round(3))