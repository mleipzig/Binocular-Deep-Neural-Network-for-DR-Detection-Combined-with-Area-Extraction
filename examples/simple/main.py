import argparse
import torch
from examples.simple.our_model import Classifier
from examples.simple.trainer import Trainer
from tensorboardX import SummaryWriter
from .image_utils import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
PATH = "/newNAS/Workspaces/DRLGroup/xiangyuliu/images"
EXCEL = "/newNAS/Workspaces/DRLGroup/xiangyuliu/label.xlsx"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.2 ** (epoch // 30))
    if lr <= 1e-5:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    classifier = Classifier(args).to(device)
    trainer = Trainer(classifier, args)
    dataset = DataLoader(PATH, EXCEL)
    logger = SummaryWriter('log:' + str(args.batch_size) + "--" + str(args.image_size))
    j = 0
    for i in range(args.epoch):
        adjust_learning_rate(trainer.optimizer, i, args)
        batch, labels = dataset.sample_minibatch(args.batch_size)
        loss = trainer.train_a_batch_binary(batch, labels)
        print(loss)
        logger.add_scalar("loss", loss, i)
        if i % args.eval_freq == args.eval_freq - 1:
            test_batch, test_label = dataset.sample_minibatch(100)
            print("----test accuracy")
            test_accuracy = trainer.evaluate_binary(test_batch, test_label)
            print("----train accuracy")
            train_accuracy = trainer.evaluate_binary(batch, labels)
            print("test_accuracy:", test_accuracy, " train_accuracy:", train_accuracy)
            logger.add_scalar("accuracy", test_accuracy, j)
            j += 1


# Todo: (1)data preprocess(add more samples and normalize) (2)partition the data set (3)multiprocess (4) try gpu version (5) visualize the figure
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
