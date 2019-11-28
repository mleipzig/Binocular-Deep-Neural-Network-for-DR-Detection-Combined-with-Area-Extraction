import argparse
import torch

from examples.simple.main import modify_labels
from examples.simple.our_model import Classifier
from examples.simple.trainer import Trainer
from examples.simple.dataset import CustomDataset
import numpy as np

device = torch.device('cpu')
path_list = ["/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3.npy",
            "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4.npy"]
def main(args):
    save_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/EfficientNet-PyTorch/examples/simple/logs/four/3/load locallyFalse-squeezeFalse/300-48-1e-10/run1/param_1700.pt"
    classifier = Classifier(args).to(device)
    classifier.load_state_dict(torch.load(save_path))
    trainer = Trainer(classifier, args)

    train_data = CustomDataset(path_list, img_size=args.image_size, model_type=args.model_type)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)
    train_accuracy_list = []
    for i in range(args.epoch):
        # the batch and labels are both tensors
        for image_batch, labels in train_loader:
            labels = modify_labels(labels)
            # we need the label of shape (?, 2)
            train_accuracy, train_loss = trainer.evaluate(image_batch, labels)
            print("train accuracy:", train_accuracy)
            train_accuracy_list.append(train_accuracy)
    print("final accuracy", np.mean(np.array(train_accuracy_list)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--final_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_local", default=False, action="store_true")
    parser.add_argument("--squeeze", default=False, action="store_true")
    parser.add_argument("--image_size", default=300, type=int)
    parser.add_argument("--model_type", default="four", type=str)
    parser.add_argument("--model_scale", default=3, type=int)
    args = parser.parse_args()
    main(args)
