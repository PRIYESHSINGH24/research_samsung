import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Paper: LeNet-5 Teacher (61,706 params)
    2 conv layers + 3 FC layers
    "Half the number of filters in conv layers" = Student
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x, temperature=1.0):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits / temperature
    
    def get_final_weights(self):
        return self.fc3.weight.data.clone()


class LeNet5Half(nn.Module):
    """
    Paper: LeNet-5-Half Student (35,820 params)
    Half conv filters: 6→3, 16→8
    FC layers SAME: 120, 84
    """
    def __init__(self, num_classes=10):
        super(LeNet5Half, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=5)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x, temperature=1.0):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits / temperature
    
    def get_final_weights(self):
        return self.fc3.weight.data.clone()


if __name__ == "__main__":
    print("="*45)
    print("  LENET VERIFICATION")
    print("="*45)
    
    teacher = LeNet5()
    student = LeNet5Half()
    
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    
    print(f"  Teacher Params : {t_params:,}")
    print(f"  Paper Target   : 61,706")
    print(f"  Match          : {'✅' if t_params == 61706 else '❌'}")
    
    print(f"\n  Student Params : {s_params:,}")
    print(f"  Paper Target   : 35,820")
    print(f"  Match          : {'✅' if s_params == 35820 else '❌'}")
    
    x = torch.randn(4, 1, 32, 32)
    t_out = teacher(x)
    s_out = student(x)
    
    print(f"\n  Teacher Output : {t_out.shape}")
    print(f"  Student Output : {s_out.shape}")
    
    weights = teacher.get_final_weights()
    print(f"  Final Weights  : {weights.shape}")