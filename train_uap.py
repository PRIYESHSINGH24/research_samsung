import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

from models.lenet import LeNet5
from models.generator import UAPGenerator
from config import config


def get_mnist_test_loader():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST(
        config.DATA_DIR, train=False,
        download=False, transform=transform
    )
    return DataLoader(
        test_data, batch_size=128,
        shuffle=False, num_workers=0, pin_memory=True
    )


def fooling_rate(model, loader, uap, device):
    model.eval()
    total = 0
    fooled = 0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            
            orig_preds = model(images).argmax(1)
            
            perturbed = images + uap
            
            pert_preds = model(perturbed).argmax(1)
            
            fooled += (orig_preds != pert_preds).sum().item()
            total += images.size(0)
    
    return 100. * fooled / total


def train_uap_generator(teacher, di_images, dataset_name='MNIST'):
    print(f"\n{'='*45}")
    print(f"  UAP TRAINING — {dataset_name}")
    print(f"{'='*45}")
    print(f"  DIs Used    : {len(di_images):,}")
    
    in_channels = di_images.shape[1]
    
    # Bigger epsilon for normalized images
    # MNIST normalized range ~[-0.42, 2.82]
    # 10/255 in pixel space = bigger in normalized space
    epsilon = 0.3  
    
    generator = UAPGenerator(
        latent_dim=10,
        channels=in_channels,
        epsilon=epsilon
    ).to(config.DEVICE)
    
    print(f"  Epsilon     : {epsilon}")
    
    # CHANGE 1: Higher LR
    optimizer = optim.Adam(
        generator.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999)
    )
    
    teacher.eval()
    test_loader = get_mnist_test_loader()
    
    di_dataset = TensorDataset(di_images, torch.zeros(len(di_images)))
    di_loader = DataLoader(
        di_dataset, batch_size=64,
        shuffle=True, num_workers=0, pin_memory=True
    )
    
    epochs = 20
    
    best_fooling = 0
    best_uap = None
    
    for epoch in range(1, epochs + 1):
        generator.train()
        total_loss = 0
        n_batches = 0
        
        for di_batch, _ in tqdm(di_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            di_batch = di_batch.to(config.DEVICE)
            B = di_batch.shape[0]
            
            z = torch.randn(B, 10, device=config.DEVICE) * 2 - 1
            uaps = generator(z)
            
            perturbed = di_batch + uaps
            
            # FOOLING LOSS — Maximize entropy of predictions
            # (push predictions away from any class)
            with torch.no_grad():
                clean_preds = teacher(di_batch).argmax(1)
            
            pert_logits = teacher(perturbed)
            
            # CHANGE 2: Use Cross-Entropy with WRONG label
            # We want predictions to CHANGE from clean_preds
            # So minimize prob of clean_preds, maximize others
            
            # Negative log likelihood of clean class (we want this LOW)
            log_probs = F.log_softmax(pert_logits, dim=1)
            fool_loss = log_probs.gather(1, clean_preds.unsqueeze(1)).mean()
            # We minimize log_prob[clean_class] => makes it LOW
            # = predictions move AWAY from clean class
            
            loss = fool_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Evaluate fooling rate
        with torch.no_grad():
            generator.eval()
            # Average UAP over many samples for stable test
            z = torch.randn(50, 10, device=config.DEVICE) * 2 - 1
            test_uaps = generator(z)
            test_uap = test_uaps.mean(0, keepdim=True)
        
        rate = fooling_rate(teacher, test_loader, test_uap, config.DEVICE)
        
        if rate > best_fooling:
            best_fooling = rate
            best_uap = test_uap.clone()
        
        print(f"  Epoch [{epoch:2d}/{epochs}] "
              f"Loss: {total_loss/n_batches:.4f} | "
              f"Fooling Rate: {rate:.2f}% | "
              f"Best: {best_fooling:.2f}%")
    
    print(f"\n{'='*45}")
    print(f"  BEST FOOLING RATE: {best_fooling:.2f}%")
    print(f"  Paper Target     : 96.45%")
    print(f"  Paper CI Result  : 91.10%")
    print(f"{'='*45}")
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'generator_state': generator.state_dict(),
        'best_uap': best_uap.cpu(),
        'fooling_rate': best_fooling
    }, f'{config.CHECKPOINT_DIR}uap_mnist.pth')
    
    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    with torch.no_grad():
        z = torch.randn(5, 10, device=config.DEVICE) * 2 - 1
        sample_uaps = generator(z)
    
    for i in range(5):
        uap_img = sample_uaps[i].squeeze().cpu().numpy()
        uap_img = (uap_img - uap_img.min()) / (uap_img.max() - uap_img.min() + 1e-8)
        axes[i].imshow(uap_img, cmap='gray')
        axes[i].set_title(f'UAP {i+1}')
        axes[i].axis('off')
    
    plt.suptitle(f'Generated UAPs — {dataset_name} (Fooling: {best_fooling:.2f}%)')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}uap_mnist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ UAPs saved!")
    
    return best_fooling


if __name__ == "__main__":
    print("="*50)
    print("  SECTION 4.4 — UAP GENERATION")
    print("="*50)
    
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()
    print(f"\n  ✅ Teacher Loaded")
    
    di_data = torch.load(
        f'{config.CHECKPOINT_DIR}data_impressions.pth',
        map_location='cpu'
    )
    di_images = di_data['di_images']
    print(f"  ✅ DIs Loaded: {di_images.shape}")
    
    fooling = train_uap_generator(teacher, di_images, 'MNIST')
    
    print(f"\n{'='*50}")
    print(f"  TABLE 10 — UAP RESULTS")
    print(f"{'='*50}")
    print(f"  Method           Fooling Rate")
    print(f"  ─────────────────────────────")
    print(f"  CI (Paper)       91.10%")
    print(f"  DI (Paper)       96.45%")
    print(f"  DI (Ours)        {fooling:.2f}%")
    print(f"{'='*50}")