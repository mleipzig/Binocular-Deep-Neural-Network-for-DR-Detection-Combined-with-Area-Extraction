import json
import argparse
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
import numpy as np
from tensorboardX import SummaryWriter

PATH = "/newNAS/Workspaces/DRLGroup/xiangyuliu/images"
EXCEL = "/newNAS/Workspaces/DRLGroup/xiangyuliu/label.xlsx"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class Classifier(torch.nn.Module):
    def __init__(self, image_base_path, label_path, args):
        super(Classifier, self).__init__()
        self.from_scratch = args.from_scrtch
        self.learing_rate = args.lr
        self.image_base_path = image_base_path
        self.label_path = label_path
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)
        self.image_label_dict = json.load(open("data_dict.json", "r"))
        self.tfms = transforms.Compose(
            [transforms.Resize(size=(args.image_size, args.image_size)), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.total_sample_num = len(self.image_label_dict.values())
        self.criteria = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learing_rate)
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

    def sample_minibatch(self, batch_size):
        image_array = []
        label_array = []
        binary_label = []
        raw_image = np.random.choice(list(self.image_label_dict.keys()), size=batch_size + 5)
        for image_path in raw_image:
            image = os.path.splitext(image_path)[0]
            image = os.path.split(image)[1]
            if os.path.exists(os.path.join(PATH, image)):
                image = self.tfms(Image.open(os.path.join(PATH, image))).unsqueeze(0)
                image_array.append(image)
                label_array.append(self.image_label_dict[image_path])
                binary_label.append(0 if self.image_label_dict[image_path] < 0.5 else 1)
            if len(image_array) == batch_size:
                break
        minibatch = torch.stack([image_array[i][0] for i in range(len(image_array))], dim=0)
        label_tensor = torch.tensor(label_array, dtype=torch.long)
        binary_label_tensor = torch.tensor(binary_label, dtype=torch.long)
        return minibatch, torch.cat((label_tensor.view(1, -1), binary_label_tensor.view(1, -1)), dim=0)

    def load_label_dict(self):
        worksheet = pd.read_excel(self.label_path, sheet_name="Sheet2")
        return worksheet.set_index("path").to_dict()['level']

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

    def train_a_batch_binary(self, batch, labels):
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.forward(batch)
        loss = self.criteria(outputs[:, 5:], labels[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_a_batch_four(self, batch, labels):
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.forward(batch)
        loss = self.criteria(outputs[:, 0:5], labels[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_binary(self, batch, labels):
        batch = batch.to(device)
        labels = labels.to(device)
        self.model.eval()
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            for i in range(batch_size):
                outputs = self.model(batch[i].view((1,) + batch[i].shape))
                output_1 = outputs[0][5:]
                output_2 = outputs[0][0:5]
                healthy = torch.argmax(output_2)
                print('-----')
                print("label:", labels[1][i], "predict:", healthy)
                if healthy == labels[1][i]:
                    accuracy += 1
        return accuracy / batch_size


class Preprocess():
    def __init__(self):
        self.label_dict = json.load(open("data_dict.json", "r"))

    def calculate_num_per_kind(self):
        return json.load(open("num_per_kind.json", "r"))


def main(args):
    classifier = Classifier(PATH, EXCEL, args).to(device)
    logger = SummaryWriter('log')
    j = 0
    for i in range(args.epoch):
        batch, labels = classifier.sample_minibatch(args.batch_size)
        loss = classifier.train_a_batch_binary(batch, labels)
        print(loss)
        logger.add_scalar("loss", loss, i)
        if i % args.eval_freq == args.eval_freq - 1:
            test_batch, test_label = classifier.sample_minibatch(100)
            accuracy = classifier.evaluate_binary(test_batch, test_label)
            print(accuracy)
            logger.add_scalar("accuracy", accuracy, j)
            j += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--epoch", default=10000, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--from_scratch", default=False, action="store_true")
    parser.add_argument("--image_size", default=656, type=int)
    args = parser.parse_args()
    main(args)
