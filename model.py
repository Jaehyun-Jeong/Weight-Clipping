import torch as T
import torch.nn as nn
import torch.nn.functional as F


class FCNLeakyReLU(nn.Module):

    def __init__(
        self,
        n_obs=10,
        n_outputs=10,
        n_hidden_units=300
    ):

        super().__init__()

        self.linear()
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.LeakyReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.LeakyReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = T.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


class NetSin(nn.Module):

    def __init__(self):

        super(NetSin, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = T.sin(x)
        x = self.conv2(x)
        x = T.sin(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = T.flatten(x, 1)
        x = self.fc1(x)
        x = T.sin(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


class Block(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        identity_downsample=None,
        stride: int = 1,
    ):

        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample


    def forward(
        self,
        x: T.Tensor
    ):

        identity = x  # for skip connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity  # skip connection
        x = F.relu(x)

        return x


class ResNet18(nn.Module):

    def __init__(
        self,
        image_channels: int,
        num_labels: int,
    ):

        super(ResNet18, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self.__make_layer(
            64,
            64,
            stride=1,
        )
        self.layer2 = self.__make_layer(
            64,
            128,
            stride=2,
        )
        self.layer3 = self.__make_layer(
            128,
            256,
            stride=2,
        )
        self.layer4 = self.__make_layer(
            256,
            512,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_labels)


    def identity_downsample(
        self,
        in_channels: int,
        out_channels: int,
    ):

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels)
        )


    # private method
    def __make_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(
                in_channels,
                out_channels
            )

        return nn.Sequential(
            Block(
                in_channels,
                out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            ),
            Block(
                out_channels,
                out_channels,
            )
        )


    def forward(
        self,
        x: T.Tensor,
    ):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = T.flatten(x, 1)
        x = self.fc(x)

        return x
