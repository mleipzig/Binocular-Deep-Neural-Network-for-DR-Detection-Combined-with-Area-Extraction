import json
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import efficientnet_pytorch
from efficientnet_pytorch import EfficientNet
import numpy as np

PATH = "/newNAS/Workspaces/DRLGroup/xiangyuliu/images"
EXCEL = "/newNAS/Workspaces/DRLGroup/xiangyuliu/label.xlsx"

class Classifier(torch.nn.Module):
    def __init__(self, image_base_path, label_path, lr=0.001, from_scrtch=True):
        super(Classifier, self).__init__()
        self.from_scratch = from_scrtch
        self.learing_rate = lr
        self.image_base_path = image_base_path
        self.label_path = label_path
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=5)
        self.image_label_dict = json.load(open("data_dict.json", "r"))
        self.tfms = transforms.Compose([transforms.Resize(size=(224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        self.total_sample_num = len(self.image_label_dict.values())
        self.criteria = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learing_rate)
        if not self.from_scratch:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model._fc = torch.nn.Linear(efficientnet_pytorch.utils.round_filters(1280, self.model._global_params),
                                             3)

    def sample_minibatch(self, batch_size):
        image_array = []
        label_array = []
        raw_image = np.random.choice(list(self.image_label_dict.keys()), size=batch_size + 5)
        for image_path in raw_image:
            image = os.path.splitext(image_path)[0]
            image = os.path.split(image)[1]
            if os.path.exists(os.path.join(PATH, image)):
                image = self.tfms(Image.open(os.path.join(PATH, image))).unsqueeze(0)
                image_array.append(image)
                label_array.append(self.image_label_dict[image_path])
            if len(image_array) == batch_size:
                break
        return torch.stack([image_array[i][0] for i in range(len(image_array))], dim=0), torch.tensor(label_array)

    def load_label_dict(self):
        worksheet = pd.read_excel(self.label_path, sheet_name="Sheet2")
        return worksheet.set_index("path").to_dict()['level']

    def forward(self, inputs):
        return self.model.forward(inputs)

    def train_a_batch(self, batch, labels):
        batch = batch.to(device)
        labels = labels.to(device)
        outputs = self.model.forward(batch)
        loss = self.criteria(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, batch, labels):
        batch = batch.to(device)
        labels = labels.to(device)
        self.model.eval()
        batch_size = batch.shape[0]
        accuracy = 0
        with torch.no_grad():
            for i in range(batch_size):
                outputs = self.model(batch[i].view((1,)+ batch[i].shape))
                print('-----')
                for idx in torch.topk(outputs, k=1).indices.squeeze(0).tolist():
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    print(idx, labels[i].item(), prob)
                    if idx == labels[i]:
                        accuracy += 1
        return accuracy / batch_size

class Preprocess():
    def __init__(self):
        self.label_dict = json.load(open("data_dict.json", "r"))
    def calculate_num_per_kind(self):
        return json.load(open("num_per_kind.json", "r"))


if __name__ == '__main__':
    preprocess = Preprocess()
    preprocess.calculate_num_per_kind()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classifier = Classifier(PATH, EXCEL).to(device)
    epoch = 20
    batch_size = 40
    for i in range(epoch):
        batch, labels = classifier.sample_minibatch(batch_size)
        loss = classifier.train_a_batch(batch, labels)
        print(loss)
        if i == epoch-1:
            test_batch, test_label = classifier.sample_minibatch(100)
            print(classifier.evaluate(test_batch, test_label))
