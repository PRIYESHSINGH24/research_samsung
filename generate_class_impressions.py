import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

from models.lenet import LeNet5
from config import config


def generate_class_impressions(model, num_per_class=2400):
    """Class Impressions = one-hot target (special case of DI)"""
    print("\n" + "="*45)
    print("  GENERATING CLASS IMPRESSIONS")
    print("="*45)
    
    model.eval()
    all_images = []
    all_labels = []
    
    for class_idx in range(config.NUM_CLASSES):
        class_cis = []
        
        # One-hot target
        target = torch.zeros(1, config.NUM_CLASSES, device=config.DEVICE)
        target[0, class_idx] = 1.0
        
        pbar = tqdm(
            range(0, num_per_class, config.DI_BATCH_SIZE),
            desc=f"  CI Class {class_idx}"
        )
        
        for start in pbar:
            end = min(start + config.DI_BATCH_SIZE, num_per_class)
            B = end - start
            
            # Random init
            ci_batch = torch.randn(
                B, config.IN_CHANNELS,
                config.IMAGE_SIZE, config.IMAGE_SIZE,
                device=config.DEVICE
            )
            ci_batch.requires_grad_(True)
            
            target_batch = target.expand(B, -1)
            optimizer = torch.optim.Adam([ci_batch], lr=config.DI_LR)
            
            for it in range(config.DI_ITERATIONS):
                optimizer.zero_grad()
                logits = model(ci_batch, temperature=config.TEMPERATURE)
                pred_softmax = F.softmax(logits, dim=1)
                loss = -torch.mean(
                    torch.sum(target_batch * torch.log(pred_softmax + 1e-8), dim=1)
                )
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    ci_batch.clamp_(-2.5, 2.5)
            
            class_cis.append(ci_batch.detach().cpu())
        
        class_tensor = torch.cat(class_cis, dim=0)
        all_images.append(class_tensor)
        all_labels.extend([class_idx] * num_per_class)
        print(f"  ✅ Class {class_idx}: {class_tensor.shape[0]} CIs done")
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"\n  Total CIs: {all_images.shape}")
    
    save_path = f'{config.CHECKPOINT_DIR}class_impressions.pth'
    torch.save({
        'ci_images': all_images,
        'ci_labels': all_labels,
    }, save_path)
    print(f"  ✅ Saved: {save_path}")
    
    return all_images, all_labels


if __name__ == "__main__":
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()
    
    cis, labels = generate_class_impressions(teacher, num_per_class=2400)