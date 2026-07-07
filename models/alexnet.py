import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    Modified AlexNet for 32x32 CIFAR-10 (Paper Section 4.1.1)
    5 conv layers with BatchNorm, Pool on 1,2,5
    3 FC layers — smaller than standard AlexNet
    Target: ~1.65M params
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1 + BN + Pool
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 16×16
            
            # Conv2 + BN + Pool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 8×8
            
            # Conv3 + BN
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv4 + BN
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv5 + BN + Pool
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 4×4
        )
        # Feature: 128 × 4 × 4 = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Linear(100, num_classes)

    def forward(self, x, temperature=1.0):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        logits = self.final_layer(x)
        return logits / temperature

    def get_final_weights(self):
        return self.final_layer.weight.data.clone()


class AlexNetHalf(nn.Module):
    """
    AlexNet-Half: half conv filters, adjusted FC
    Target: ~723K params
    """
    def __init__(self, num_classes=10):
        super(AlexNetHalf, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 64 × 4 × 4 = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(384, 48),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Linear(48, num_classes)

    def forward(self, x, temperature=1.0):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        logits = self.final_layer(x)
        return logits / temperature

    def get_final_weights(self):
        return self.final_layer.weight.data.clone()

if __name__ == "__main__":
    print("="*45)
    print("  ALEXNET VERIFICATION")
    print("="*45)
    
    teacher = AlexNet()
    student = AlexNetHalf()
    
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    
    print(f"  AlexNet Params      : {t_params:,}")
    print(f"  Paper Target        : ~1,650,000")
    
    print(f"\n  AlexNet-Half Params : {s_params:,}")
    print(f"  Paper Target        : ~723,000")
    
    x = torch.randn(4, 3, 32, 32)
    t_out = teacher(x)
    s_out = student(x)
    
    print(f"\n  AlexNet Output      : {t_out.shape}")
    print(f"  AlexNet-Half Output : {s_out.shape}")
    
    weights = teacher.get_final_weights()
    print(f"  Final Weights       : {weights.shape}")
    
    # Layer-wise params
    print(f"\n  --- Layer-wise (Teacher) ---")
    for name, param in teacher.named_parameters():
        print(f"  {name:<35s} {param.numel():>10,}")