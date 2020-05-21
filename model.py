import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = DoubleConv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = DoubleConv2d(512, 1024, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(self.pooling(out1))
        out3 = self.conv3(self.pooling(out2))
        out4 = self.conv4(self.pooling(out3))
        x = self.conv5(self.pooling(out4))
        return out1, out2, out3, out4, x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv1 = UpConv2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = UpConv2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = UpConv2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = UpConv2d(128, 64, kernel_size=2, stride=2)

        self.conv1 = DoubleConv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(128, 64, kernel_size=3, padding=1)

        self.conv1x1 = Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, out1, out2, out3, out4, x):
        x = self.upconv1(x)
        x = torch.cat([x, out4], dim=1)
        x = self.conv1(x)

        x = self.upconv2(x)
        x = torch.cat([x, out3], dim=1)
        x = self.conv2(x)

        x = self.upconv3(x)
        x = torch.cat([x, out2], dim=1)
        x = self.conv3(x)

        x = self.upconv4(x)
        x = torch.cat([x, out1], dim=1)
        x = self.conv4(x)

        x = self.conv1x1(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x)
        x = self.decoder(out1, out2, out3, out4, x)
        return x
