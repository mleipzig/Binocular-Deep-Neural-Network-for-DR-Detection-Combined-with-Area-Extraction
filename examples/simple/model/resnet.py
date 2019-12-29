'''
In this file we implement a rather simple resnet
'''
import argparse
import torch

from examples.simple.model.main import path_list, adjust_learning_rate
from examples.simple.model.our_model import ResNet
from examples.simple.model.trainer import Trainer, device
from tensorboardX import SummaryWriter
from examples.simple.model.dataset import CustomDataset
import numpy as np
from pathlib import Path
import os


def modify_labels(labels):
    old_label_np = labels.numpy()
    new_label_np = np.zeros(shape=old_label_np.shape)
    for i in range(old_label_np.shape[0]):
        new_label_np[i] = 0 if old_label_np[i] < 0.5 else 1
    labels = np.concatenate([old_label_np.reshape(1, -1), new_label_np.reshape(1, -1)], axis=0)
    return torch.tensor(labels, dtype=torch.long)


def main(args):
    model_dir = Path('./resnet') / (str(args.image_size) + "-" + str(args.batch_size) + "-" + str(args.final_lr))
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

    # save_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/EfficientNet-PyTorch/examples/simple/logs/four/3/load locallyFalse-squeezeFalse/300-48-1e-10/run1/param_400.pt"
    classifier = ResNet(block=None, layers=[2, 2, 2], num_classes=4).to(device)
    print(classifier)
    # if args.load_local:
    #     classifier.load_state_dict(torch.load(save_path))
    #     args.lr = args.final_lr
    trainer = Trainer(classifier, args)
    train_data = CustomDataset(path_list, img_size=args.image_size, model_type=args.model_type)
    train_data.test_label = modify_labels(train_data.test_label)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=2)
    iter = 0
    j = 0
    for i in range(args.epoch):
        # the batch and labels are both tensors
        for image_batch, labels in train_loader:
            tmp_lr = adjust_learning_rate(trainer.optimizer, i, args)
            labels = modify_labels(labels)
            # we need the label of shape (?, 2)
            loss = trainer.train(image_batch, labels)
            print(iter, loss)
            logger.add_scalar("train_loss", loss, iter)
            if iter % args.eval_freq == args.eval_freq - 1 or iter == 0:
                print("----test results")
                test_accuracy, test_loss = trainer.evaluate(train_data.test_batch, train_data.test_label)
                print("----train results")
                train_accuracy, train_loss = trainer.evaluate(image_batch, labels)
                logger.add_scalar("test_loss", test_loss, j)
                logger.add_scalar("test_accuracy", test_accuracy, j)
                logger.add_scalar("train_accuracy", train_accuracy, j)
                logger.add_scalar("learning rate", tmp_lr, j)
                print("train accuracy:", train_accuracy, " test accuracy:", test_accuracy)
                print("train loss:", train_loss, " test loss:", test_loss)
                j += 1
            if iter % args.save_freq == 0:
                print("saving model")
                torch.save(classifier.state_dict(), log_dir / ("param_%i.pt" % iter))
            iter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_local", default=False, action="store_true")
    parser.add_argument("--image_size", default=300, type=int)
    parser.add_argument("--model_type", default="univ_net", type=str)
    args = parser.parse_args()
    main(args)
