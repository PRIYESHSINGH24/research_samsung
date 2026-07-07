import torch
import torch.nn as nn


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# VGG Configs
vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = make_layers(vgg_cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.final_layer = nn.Linear(512, num_classes)

    def forward(self, x, temperature=1.0):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        logits = self.final_layer(x)
        return logits / temperature

    def get_final_weights(self):
        return self.final_layer.weight.data.clone()


def VGG19(num_classes=10):
    return VGG('VGG19', num_classes)

def VGG11(num_classes=10):
    return VGG('VGG11', num_classes)


if __name__ == "__main__":
    print("="*45)
    print("VGG VERIFICATION")
    print("="*45)

    teacher = VGG19()
    student = VGG11()

    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())

    print(f"VGG-19 Params : {t_params:,}")
    print(f"VGG-11 Params : {s_params:,}")

    x = torch.randn(4, 3, 32, 32)
    t_out = teacher(x)
    s_out = student(x)

    print(f"\nVGG-19 Output : {t_out.shape}")
    print(f"VGG-11 Output : {s_out.shape}")

    weights = teacher.get_final_weights()
    print(f"Final Weights : {weights.shape}")