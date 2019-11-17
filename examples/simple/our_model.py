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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Classifier(torch.nn.Module):
    def __init__(self, image_base_path, label_path, args):
        super(Classifier, self).__init__()
        self.from_scratch = args.from_scratch
        self.image_base_path = image_base_path
        self.label_path = label_path
        self.model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5)
        self.image_label_dict = json.load(open("data_dict.json", "r"))
        self.tfms = transforms.Compose([transforms.Resize(size=(args.image_size, args.image_size)),
                                        transforms.ToTensor(),])
        self.total_sample_num = len(self.image_label_dict.values())
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

class Trainer():
    def __init__(self, model, args):
        self.learing_rate = args.lr
        self.model = model
        self.criteria = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learing_rate)
    def train_a_batch_binary(self, batch, labels):
        self.model.train()
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.model(batch)
        loss = self.criteria(outputs[:, 5:], labels[1])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def train_a_batch_four(self, batch, labels):
        self.model.train()
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.model(batch)
        loss = self.criteria(outputs[:, 0:5], labels[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate_binary(self, batch, labels):
        self.model.eval()
        batch = batch.to(device)
        labels = labels.to(device)
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            for i in range(batch_size):
                outputs = self.model(batch[i].view((1,) + batch[i].shape))
                output_2 = outputs[0][5:]
                healthy = torch.argmax(output_2)
                print('-----')
                print(" label:", labels[1][i].item(), " predict:", healthy.item(), " prob:", torch.max(torch.softmax(output_2, dim=0)).item())
                if healthy == labels[1][i]:
                    accuracy += 1
        return accuracy / batch_size

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 30))
    if lr <= 1e-5:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    classifier = Classifier(PATH, EXCEL, args).to(device)
    trainer = Trainer(classifier, args)
    logger = SummaryWriter('log:'+str(args.batch_size)+ "--" + str(args.image_size))
    j = 0
    for i in range(args.epoch):
        adjust_learning_rate(trainer.optimizer, i, args)
        batch, labels = classifier.sample_minibatch(args.batch_size)
        loss = trainer.train_a_batch_binary(batch, labels)
        print(loss)
        logger.add_scalar("loss", loss, i)
        if i % args.eval_freq == args.eval_freq - 1:
            test_batch, test_label = classifier.sample_minibatch(100)
            print("----test accuracy")
            test_accuracy = trainer.evaluate_binary(test_batch, test_label)
            print("----train accuracy")
            train_accuracy = trainer.evaluate_binary(batch, labels)
            print("test_accuracy:", test_accuracy, " train_accuracy:", train_accuracy)
            logger.add_scalar("accuracy", test_accuracy, j)
            j += 1

# Todo: (1)data preprocess(add more samples and normalize) (2)partition the data set (3)multiprocess (4) try gpu version (5) visualize the figure
# Todo: modify the batchsize
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--epoch", default=10000, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--from_scratch", default=True, action="store_false")
    parser.add_argument("--image_size", default=300, type=int)
    args = parser.parse_args()
    main(args)
