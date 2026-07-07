import torch
from models.lenet import LeNet5, LeNet5Half
from models.vgg import VGG19, VGG11
from models.resnet import ResNet18, ResNet18Half

def show_params(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    
    print(f"\n  {name}:")
    print(f"    Total Params       : {total:,}")
    print(f"    Trainable Params   : {trainable:,}")
    print(f"    Non-Trainable      : {non_trainable:,}")

print("="*50)
print("  ALL MODEL PARAMETERS")
print("="*50)

print("\n  --- MNIST Models ---")
show_params(LeNet5(), "LeNet-5 (Teacher)")
show_params(LeNet5Half(), "LeNet-5-Half (Student)")

print("\n  --- CIFAR-10 Models ---")
show_params(VGG19(), "VGG-19 (Teacher)")
show_params(VGG11(), "VGG-11 (Student)")
show_params(ResNet18(), "ResNet-18 (Teacher)")
show_params(ResNet18Half(), "ResNet-18-Half (Student)")