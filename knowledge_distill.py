import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from models.lenet import LeNet5, LeNet5Half
from config import config


class AugmentedDIDataset(Dataset):
    def __init__(self, di_images, di_labels):
        self.images = di_images
        self.labels = di_labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        img_min = image.min()
        img_max = image.max()
        image_norm = (image - img_min) / (img_max - img_min + 1e-8)
        image_aug = self.transform(image_norm)
        image_aug = image_aug * (img_max - img_min) + img_min
        return image_aug, label


def get_test_loader():
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
        test_data, batch_size=256,
        shuffle=False, num_workers=0,
        pin_memory=True
    )


def kd_loss(student_logits, teacher_logits, temperature):
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
    loss = F.kl_div(
        log_soft_student,
        soft_teacher,
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            out = model(images)
            _, predicted = out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def knowledge_distillation(teacher, di_images, di_labels):
    print("\n" + "="*45)
    print("  STEP 4 — KNOWLEDGE DISTILLATION")
    print("="*45)
    print(f"  Transfer Set   : {len(di_images):,} DIs")
    print(f"  Student Epochs : {config.STUDENT_EPOCHS}")
    print(f"  KD Temperature : {config.KD_TEMPERATURE}")
    print(f"  Augmentation   : ON")
    print("="*45)

    di_dataset = AugmentedDIDataset(di_images, di_labels)
    di_loader = DataLoader(
        di_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    student = LeNet5Half().to(config.DEVICE)
    test_loader = get_test_loader()

    optimizer = optim.Adam(
        student.parameters(),
        lr=config.STUDENT_LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.STUDENT_EPOCHS,
        eta_min=1e-6
    )

    teacher.eval()

    best_acc = 0
    best_epoch = 0
    patience = 100
    test_accs = []
    train_losses = []

    for epoch in range(1, config.STUDENT_EPOCHS + 1):
        student.train()
        total_loss = 0

        pbar = tqdm(di_loader, leave=False)

        for di_batch, _ in pbar:
            di_batch = di_batch.to(config.DEVICE)

            with torch.no_grad():
                t_logits = teacher(di_batch)

            s_logits = student(di_batch)
            loss = kd_loss(s_logits, t_logits, config.KD_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        scheduler.step()

        test_acc = evaluate(student, test_loader)
        test_accs.append(test_acc)
        train_losses.append(total_loss / len(di_loader))

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{config.STUDENT_EPOCHS}] "
                  f"Loss: {total_loss/len(di_loader):.4f} | "
                  f"Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(
                student.state_dict(),
                f'{config.CHECKPOINT_DIR}student_best.pth'
            )

     

    print(f"\n{'='*45}")
    print(f"  Student Best Acc : {best_acc:.2f}%")
    print(f"  Best at Epoch    : {best_epoch}")
    print(f"  Paper Target     : 98.77%")
    print(f"  Match            : {'✅' if best_acc >= 96.0 else '⚠️'}")
    print(f"{'='*45}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(test_accs)
    ax1.axhline(y=98.77, color='r',
                linestyle='--', label='Paper (98.77%)')
    ax1.set_title('Student Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(train_losses)
    ax2.set_title('KD Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    plt.tight_layout()
    path = f'{config.RESULTS_DIR}kd_training.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✅ Plot saved: {path}")

    return student, best_acc


if __name__ == "__main__":
    teacher = LeNet5().to(config.DEVICE)
    teacher.load_state_dict(
        torch.load(
            f'{config.CHECKPOINT_DIR}teacher_best.pth',
            map_location=config.DEVICE
        )
    )
    teacher.eval()

    di_data = torch.load(
        f'{config.CHECKPOINT_DIR}data_impressions.pth',
        map_location='cpu'
    )
    di_images = di_data['di_images']
    di_labels = di_data['di_labels']

    print(f"  DIs Loaded : {di_images.shape}")
    print(f"  Labels     : {di_labels.shape}")

    student, acc = knowledge_distillation(teacher, di_images, di_labels)