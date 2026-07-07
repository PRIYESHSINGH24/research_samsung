import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models.vgg import VGG19
from config import config


def check_di_quality():
    print("="*50)
    print("  DI QUALITY DIAGNOSIS")
    print("="*50)

    # Load teacher
    teacher = VGG19().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}vgg19_cifar_teacher.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    # Load DIs
    di_data = torch.load(
        f'{config.CHECKPOINT_DIR}vgg19_cifar_dis.pth',
        map_location='cpu'
    )
    di_images = di_data['di_images']
    di_labels = di_data['di_labels']

    print(f"\n  Total DIs: {di_images.shape}")
    print(f"  DI Range : [{di_images.min():.2f}, {di_images.max():.2f}]")
    print(f"  DI Mean  : {di_images.mean():.4f}")
    print(f"  DI Std   : {di_images.std():.4f}")

    # Check: Teacher predicts DIs correctly?
    print(f"\n  Testing if Teacher recognizes DIs...")
    
    correct = 0
    total = 0
    confidences = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(di_images), batch_size):
            batch = di_images[i:i+batch_size].to(config.DEVICE)
            labels = di_labels[i:i+batch_size].to(config.DEVICE)
            
            logits = teacher(batch)
            probs = F.softmax(logits, dim=1)
            
            max_probs, predicted = probs.max(1)
            confidences.extend(max_probs.cpu().numpy().tolist())
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    teacher_acc_on_dis = 100. * correct / total
    avg_confidence = np.mean(confidences) * 100
    
    print(f"\n  Teacher Acc on DIs : {teacher_acc_on_dis:.2f}%")
    print(f"  Avg Confidence     : {avg_confidence:.2f}%")
    
    print("\n" + "="*50)
    print("  DIAGNOSIS:")
    print("="*50)
    
    if teacher_acc_on_dis > 90:
        print("  ✅ DIs are GOOD — Teacher recognizes them")
        print("  → Problem is in KD training")
    elif teacher_acc_on_dis > 50:
        print("  ⚠️ DIs are MEDIUM quality")
        print("  → DIs aur KD dono mein issue ho sakta hai")
    else:
        print("  ❌ DIs are BAD QUALITY")
        print("  → Teacher khud DIs nahi pehchanta")
        print("  → KD tweak useless hai")
        print("  → DIs dobara generate karne padenge")
    
    # Visualize some DIs
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for cls in range(10):
        mask = (di_labels == cls)
        idx = mask.nonzero()[0][0].item()
        di = di_images[idx]
        
        # Denormalize for display
        di_img = di.permute(1, 2, 0).numpy()
        di_img = (di_img - di_img.min()) / (di_img.max() - di_img.min() + 1e-8)
        
        ax = axes[cls // 5][cls % 5]
        ax.imshow(di_img)
        ax.axis('off')
        ax.set_title(f'Class {cls}', fontsize=10)
    
    plt.suptitle('CIFAR Data Impressions (one per class)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/cifar_di_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ DI visualization saved to results/cifar_di_quality.png")
    

if __name__ == "__main__":
    check_di_quality()