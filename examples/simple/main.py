import argparse
import torch
from examples.simple.our_model import Classifier
from examples.simple.trainer import Trainer
from tensorboardX import SummaryWriter
from examples.simple.dataset import CustomDataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH = "/newNAS/Workspaces/DRLGroup/xiangyuliu/images"
EXCEL = "/newNAS/Workspaces/DRLGroup/xiangyuliu/label.xlsx"
path_list = ["/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4_clahe.npy"]

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.2 ** (epoch // 30))
    if lr <= 1e-4:
        lr = 1e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def modify_labels(labels):
    old_label_np = labels.numpy()
    new_label_np = np.zeros(shape=old_label_np.shape)
    for i in range(old_label_np.shape[0]):
        new_label_np[i] = 0 if old_label_np[i] < 0.5 else 1
    labels = np.concatenate([old_label_np.reshape(1, -1), new_label_np.reshape(1, -1)], axis=0)
    return torch.tensor(labels, dtype=torch.long)


def main(args):
    classifier = Classifier(args).to(device)
    trainer = Trainer(classifier, args)
    train_data = CustomDataset(path_list, img_size=args.image_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    logger = SummaryWriter('log:' + str(args.batch_size) + "--" + str(args.image_size))
    iter = 0
    for i in range(args.epoch):
        for image_batch, labels in train_loader:
            adjust_learning_rate(trainer.optimizer, iter, args)
            labels = modify_labels(labels)
            # we need the label of shape (?, 2)
            loss = trainer.train_a_batch_binary(image_batch, labels)
            print(loss)
            logger.add_scalar("loss", loss, i)
            if iter % args.eval_freq == args.eval_freq - 1 or iter == 0:
                print("----test train results")
                train_accuracy = trainer.evaluate_binary(image_batch, labels)
                print("train_accuracy:", train_accuracy)
            iter += 1


# Todo: (1)data preprocess(add more samples and normalize) (2)partition the data set (3)multiprocess (4) try gpu version (5) visualize the figure
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--from_scratch", default=True, action="store_false")
    parser.add_argument("--image_size", default=300, type=int)
    args = parser.parse_args()
    main(args)
