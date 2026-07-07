import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from models.lenet import LeNet5
from config import config


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=False,
        transform=transform
    )
    test_data = datasets.MNIST(
        root=config.DATA_DIR,
        train=False,
        download=False,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train Samples : {len(train_data):,}")
    print(f"Test Samples  : {len(test_data):,}")
    
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def train_teacher():
    print("\n" + "="*45)
    print("  STEP 1 — TEACHER TRAINING")
    print("="*45)
    print(f"  Device  : {config.DEVICE}")
    print(f"  Epochs  : {config.TEACHER_EPOCHS}")
    print(f"  LR      : {config.TEACHER_LR}")
    print("="*45 + "\n")
    
    train_loader, test_loader = get_data_loaders()
    
    model     = LeNet5(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TEACHER_LR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 80], gamma=0.1
    )
    
    best_acc   = 0
    train_accs = []
    test_accs  = []
    
    for epoch in range(1, config.TEACHER_EPOCHS + 1):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0
        
        pbar = tqdm(train_loader,
                   desc=f"Epoch [{epoch:3d}/{config.TEACHER_EPOCHS}]")
        
        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss/len(train_loader):.4f}',
                'Acc' : f'{100.*correct/total:.2f}%'
            })
        
        test_acc = evaluate(model, test_loader)
        scheduler.step()
        
        train_accs.append(100.*correct/total)
        test_accs.append(test_acc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{config.TEACHER_EPOCHS}] "
                  f"Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(
                model.state_dict(),
                f'{config.CHECKPOINT_DIR}teacher_best.pth'
            )
            print(f"  ✅ Best Saved! Acc: {best_acc:.2f}%")
    
    print(f"\n{'='*45}")
    print(f"  Best Test Acc : {best_acc:.2f}%")
    print(f"  Paper Target  : 99.34%")
    print(f"{'='*45}")
    
    plt.figure(figsize=(10, 4))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs,  label='Test Accuracy')
    plt.axhline(y=99.34, color='r', linestyle='--',
                label='Paper Target (99.34%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Teacher Training — MNIST')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{config.RESULTS_DIR}teacher_training.png', dpi=150)
    plt.close()
    print(f"  ✅ Plot saved!")
    
    return model, best_acc


if __name__ == "__main__":
    model, acc = train_teacher()