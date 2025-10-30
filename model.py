import math as ma

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class FCNLeakyReLU(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden_units=300):

        super().__init__()

        self.linear1 = nn.Linear(n_inputs, n_hidden_units)
        self.leaky_relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(n_hidden_units, n_hidden_units // 2)
        self.leaky_relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(n_hidden_units // 2, n_outputs)

        self.weight_init()

    def forward(self, x):

        x = self.linear1(x)
        x = self.leaky_relu1(x)
        x = self.linear2(x)
        x = self.leaky_relu2(x)
        x = self.linear3(x)

        return x

    # Initialization using uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
    def weight_init(self):

        weights = [
            self.linear1.weight,
            self.linear2.weight,
            self.linear3.weight,
        ]
        biases = [
            self.linear1.bias,
            self.linear2.bias,
            self.linear3.bias,
        ]
        fan_ins = [
            self.linear1.weight.shape[1],
            self.linear2.weight.shape[1],
            self.linear3.weight.shape[1],
        ]

        len_layers = len(weights)

        with T.no_grad():
            for weight, bias, fan_in in zip(weights, biases, fan_ins):
                weight.uniform_(-1 / ma.sqrt(fan_in), 1 / ma.sqrt(fan_in))
                bias.zero_()

                """ Check proper working
                print(f"upper: {1/ma.sqrt(fan_in)}")
                print(f"lower: {-1/ma.sqrt(fan_in)}")
                print(f"max: {T.max(weight)}")
                print(f"min: {T.min(weight)}")
                """


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
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x: T.Tensor):

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
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
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
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_labels)

    def identity_downsample(
        self,
        in_channels: int,
        out_channels: int,
    ):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
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
            identity_downsample = self.identity_downsample(in_channels, out_channels)

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
            ),
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
