import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, half=False):
        super(ResNet, self).__init__()

        if half:
            self.channels = [32, 64, 128, 256]
        else:
            self.channels = [64, 128, 256, 512]

        self.in_planes = self.channels[0]

        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels[0])

        self.layer1 = self._make_layer(block, self.channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.channels[3], num_blocks[3], stride=2)

        self.final_layer = nn.Linear(self.channels[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, temperature=1.0):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.final_layer(out)
        return logits / temperature

    def get_final_weights(self):
        return self.final_layer.weight.data.clone()


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, half=False)

def ResNet18Half(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, half=True)


if __name__ == "__main__":
    print("="*45)
    print("RESNET VERIFICATION")
    print("="*45)

    teacher = ResNet18()
    student = ResNet18Half()

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())

    print(f"ResNet-18 Params      : {t_params:,}")
    print(f"ResNet-18-Half Params : {s_params:,}")

    x = torch.randn(4, 3, 32, 32)
    t_out = teacher(x)
    s_out = student(x)

    print(f"\nResNet-18 Output      : {t_out.shape}")
    print(f"ResNet-18-Half Output : {s_out.shape}")

    weights = teacher.get_final_weights()
    print(f"Final Weights         : {weights.shape}")