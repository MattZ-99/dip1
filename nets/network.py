import torch
import torch.nn as nn


def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x


class draftNet(nn.Module):
    def __init__(self, model, num_classes=1000):
        super().__init__()
        self.backbone = model
        self.fc1 = nn.Linear(64, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.sg = nn.Sigmoid()
        self.ad = nn.AdaptiveAvgPool2d(1)
        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)
        x = self.backbone.conv1(x)
        x = self.ad(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sg(self.fc2(x))

        return x


class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

        self.refine = nn.Conv2d(15, 3, kernel_size=3, stride=1, padding=1)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(30, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 15, 3, padding=1, bias=False)
                                    )

        self.ad = nn.AdaptiveAvgPool2d(1)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = self.ad(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.sg(self.fc2(x))

        return x


class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model
        self.fc1 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()

        self.backbone = model

        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()

        self.backbone = model
        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

    def forward(self, x):
        x = self.conv_1(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
