import argparse
import torch
from examples.simple.our_model import Classifier
from examples.simple.trainer import Trainer
from tensorboardX import SummaryWriter
from examples.simple.dataset import CustomDataset
import numpy as np
from pathlib import Path
import os

device = torch.device('cpu')
PATH = "/newNAS/Workspaces/DRLGroup/xiangyuliu/images"
EXCEL = "/newNAS/Workspaces/DRLGroup/xiangyuliu/label.xlsx"
path_list = ["/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_0_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_1_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_2_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_3_clahe.npy",
             "/newNAS/Workspaces/DRLGroup/xiangyuliu/clahe/x_4_clahe.npy"]


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.5 ** (epoch // 10))
    if lr <= args.final_lr:
        lr = args.final_lr
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
    model_dir = Path('./logs') / args.model_type / str(args.model_scale) / (
                "load locally" + str(args.load_local) + "-squeeze" + str(args.squeeze)) / (
                        str(args.image_size) + "-" + str(args.batch_size) + "-" + str(args.final_lr))
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    log_dir = model_dir / curr_run
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    save_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/EfficientNet-PyTorch/examples/simple/logs/binary/3/300-48/run1/param_2200.pt"
    classifier = Classifier(args).to(device)
    if args.load_local:
        classifier.load_state_dict(torch.load(save_path))
        args.lr = args.final_lr
    trainer = Trainer(classifier, args)
    train_data = CustomDataset(path_list, img_size=args.image_size)
    train_data.test_label = modify_labels(train_data.test_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)
    iter = 0
    j = 0
    for i in range(args.epoch):
        # the batch and labels are both tensors
        for image_batch, labels in train_loader:
            adjust_learning_rate(trainer.optimizer, iter, args)
            labels = modify_labels(labels)
            # we need the label of shape (?, 2)
            loss = trainer.train(image_batch, labels)
            print(iter, loss)
            logger.add_scalar("train_loss", loss, iter)
            if iter % args.eval_freq == args.eval_freq - 1:
                print("----test results")
                test_accuracy, test_loss = trainer.evaluate(train_data.test_batch, train_data.test_label)
                print("----train results")
                train_accuracy, train_loss = trainer.evaluate(image_batch, labels)
                logger.add_scalar("test_loss", test_loss, j)
                logger.add_scalar("test_accuracy", test_accuracy, j)
                print("train accuracy:", train_accuracy, " test accuracy:", test_accuracy)
                print("train loss:", train_loss, " test loss:", test_loss)
                j += 1
            if iter % args.save_freq == 0:
                print("saving model")
                classifier.save(log_dir / ("param_%i.pt" % iter))
            iter += 1


# Todo: (1)data preprocess(add more samples and normalize) (2)partition the data set (3)multiprocess (4) try gpu version (5) visualize the figure
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--final_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_local", default=False, action="store_true")
    parser.add_argument("--squeeze", default=False, action="store_true")
    parser.add_argument("--image_size", default=300, type=int)
    parser.add_argument("--model_type", default="five", type=str)
    parser.add_argument("--model_scale", default=3, type=int)
    args = parser.parse_args()
    main(args)
