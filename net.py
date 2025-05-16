import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN_5x5(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 32x32x3 (CIFAR-10)
        # 3 channels (RGB), 6 filters, 5x5 kernel
        
        # 32x32x3 -> 28x28x16 (CIFAR-10)
        self.conv1 = nn.Conv2d(3, 16, 5)
        
        # 28x28x16 -> 14x14x16 (CIFAR-10)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 16 channels, 16 filters, 5x5 kernel
        # 14x14x16 -> 10x10x16 (CIFAR-10)
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        # Fully connected layers. 800->200->84->10
        self.fc1 = nn.Linear(32 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # conv1 block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 32x32 -> 16x16

            # conv2 block
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 16x16x64 -> 16x16x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # 16x16 -> 8x8

            # conv3 block
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 8x8x128 -> 8x8x256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 8x8x256 -> 8x8x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 8x8 -> 4x4

            # conv4 block
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 4x4x256 -> 4x4x512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 4x4x512 -> 4x4x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 4x4 -> 2x2
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)  # no downsampling
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))     # 32x32
        out = self.layer1(out)                    # 32x32
        out = self.layer2(out)                    # 16x16
        out = self.layer3(out)                    # 8x8
        out = self.layer4(out)                    # 4x4
        out = F.avg_pool2d(out, 4)                # 1x1
        out = out.view(out.size(0), -1)
        return self.linear(out)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # total 18 layers