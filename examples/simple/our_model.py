import torch
from torch import nn
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
import math


def GetSpecificArea(model, images):
    location = model(images)
    row, column = images[0].shape(0), images.shape(1)
    area_list = []
    for i in range(images.shape[0]):
        image = images[i]
        center_x, center_y = location[i, 0], location[i, 1]
        left_x, right_x = (center_x - 0.1) * row, (center_x + 0.1) * row
        low_y, high_y = (center_y - 0.1) * column, (center_y + 0.1) * column
        area_list.append(image[math.floor(left_x):math.floor(right_x), math.floor(low_y):math.floor(high_y)])
    return area_list


class ExtractMacula(torch.nn.Module):
    def __init__(self):
        super(ExtractMacula, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=2)

    def forward(self, inputs):
        return self.sigmoid(self.model.forward(input))


class ExtractOptic(torch.nn.Module):
    def __init__(self):
        super(ExtractOptic, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=2)

    def forward(self, inputs):
        return self.sigmoid(self.model.forward(inputs))


class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.squeeze = args.squeeze
        self.model = EfficientNet.from_pretrained(args.model_detail, num_classes=5)
        self.model_for_macula = EfficientNet.from_pretrained("effcientnet-b3", num_classes=5)
        self.model_for_optic = EfficientNet.from_pretrained("effcientnet-b3", num_classes=5)
        if self.squeeze:
            for param in self.model.parameters():
                param.requires_grad = False
        out_channels = efficientnet_pytorch.utils.round_filters(1280,
                                                                self.model._global_params) + efficientnet_pytorch.utils.round_filters(
            1280, self.model_for_macula._global_params) + efficientnet_pytorch.utils.round_filters(1280,
                                                                                                   self.model_for_optic._global_params)
        self._avg_pooling_1 = torch.nn.AdaptiveAvgPool2d(1)
        self._dropout_1 = torch.nn.Dropout(self.model._global_params.dropout_rate)
        self._fc_1 = torch.nn.Linear(out_channels, 5)
        self._avg_pooling_2 = torch.nn.AdaptiveAvgPool2d(1)
        self._dropout_2 = torch.nn.Dropout(self.model._global_params.dropout_rate)
        self._fc_2 = torch.nn.Linear(out_channels, 2)
        self._avg_pooling_3 = torch.nn.AdaptiveAvgPool2d(1)
        self._dropout_3 = torch.nn.Dropout(self.model._global_params.dropout_rate)
        self._fc_3 = torch.nn.Linear(out_channels, 4)

    def forward(self, inputs, inputs_macular, inputs_optic):
        bs = inputs.size(0)
        x = self.model.extract_features(inputs)
        x_macular = self.model_for_macula.extract_features(inputs_macular)
        x_optic = self.model_for_optic.extract_features(inputs_optic)
        x = torch.cat((x, x_macular, x_optic), dim=-1)
        x_1 = self._avg_pooling_1(x)
        x_1 = x_1.view(bs, -1)
        x_1 = self._dropout_1(x_1)
        x_1 = self._fc_1(x_1)

        x_2 = self._avg_pooling_2(x)
        x_2 = x_2.view(bs, -1)
        x_2 = self._dropout_2(x_2)
        x_2 = self._fc_2(x_2)

        x_3 = self._avg_pooling_3(x)
        x_3 = x_3.view(bs, -1)
        x_3 = self._dropout_3(x_3)
        x_3 = self._fc_3(x_3)

        x = torch.cat((x_1, x_2, x_3), dim=-1)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, 16, layers[0])
        self.layer2 = self.make_layer(ResidualBlock, 32, layers[1], 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
