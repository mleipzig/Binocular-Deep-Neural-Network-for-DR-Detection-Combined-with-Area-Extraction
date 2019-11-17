import torch
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet


class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.from_scratch = args.from_scratch
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
        if not self.from_scratch:
            for param in self.model.parameters():
                param.requires_grad = False
        out_channels = efficientnet_pytorch.utils.round_filters(1280, self.model._global_params)
        self._avg_pooling_1 = torch.nn.AdaptiveAvgPool2d(1)
        self._dropout_1 = torch.nn.Dropout(self.model._global_params.dropout_rate)
        self._fc_1 = torch.nn.Linear(out_channels, 5)
        self._avg_pooling_2 = torch.nn.AdaptiveAvgPool2d(1)
        self._dropout_2 = torch.nn.Dropout(self.model._global_params.dropout_rate)
        self._fc_2 = torch.nn.Linear(out_channels, 2)

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.model.extract_features(inputs)
        x_1 = self._avg_pooling_1(x)
        x_1 = x_1.view(bs, -1)
        x_1 = self._dropout_1(x_1)
        x_1 = self._fc_1(x_1)
        x_2 = self._avg_pooling_2(x)
        x_2 = x_2.view(bs, -1)
        x_2 = self._dropout_2(x_2)
        x_2 = self._fc_2(x_2)
        x = torch.cat((x_1, x_2), dim=-1)
        return x
