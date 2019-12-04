import argparse
import torch
import torchvision

from examples.simple.main import modify_labels
from examples.simple.our_model import Classifier
from examples.simple.trainer import Trainer, device
from examples.simple.dataset import CustomDataset
import numpy as np

path_list = None
save_path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/EfficientNet-PyTorch/examples/simple/logs/four/3/load locallyFalse-squeezeFalse/300-48-1e-10/run1/param_1700.pt"


def main(args):
    model_dict = {"resnet18": torchvision.models.resnet18(pretrained=args.pretrain),
                  "resnet34": torchvision.models.resnet34(pretrained=args.pretrain),
                  "resnet50": torchvision.models.resnet50(pretrained=args.pretrain),
                  "resnet101": torchvision.models.resnet101(pretrained=args.pretrain),
                  "resnet152": torchvision.models.resnet152(pretrained=args.pretrain),
                  "densenet121": torchvision.models.densenet121(pretrained=args.pretrain),
                  "densenet": torchvision.models.densenet161(pretrained=args.pretrain),
                  "densenet169": torchvision.models.densenet169(pretrained=args.pretrain),
                  "densenet201": torchvision.models.densenet201(pretrained=args.pretrain),
                  "wide_resnet50_2": torchvision.models.wide_resnet50_2(pretrained=args.pretrain),
                  "wide_resnet101_2": torchvision.models.wide_resnet101_2(pretrained=args.pretrain)}

    if "efficientnet" in args.model_detail:
        classifier = Classifier(args).to(device)
    else:
        classifier = model_dict[args.model_detail].to(device)
    classifier.load_state_dict(torch.load(save_path))
    trainer = Trainer(classifier, args)
    test_data = CustomDataset(path_list, img_size=args.image_size, sort_kinds=args.sort_kinds, test=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True,
                                              num_workers=2)
    true_len = 0
    total_len = 0
    output_list = []
    for image_batch, labels in test_loader:
        labels = modify_labels(labels)
        test_accuracy, _, result = trainer.evaluate(image_batch, labels)
        print("test accuracy:", test_accuracy)
        true_len += test_accuracy * image_batch.shape[0]
        total_len += image_batch.shape[0]
        output_list.append(torch.softmax(result, dim=0).numpy())
    np.save("prob.npy", np.concatenate(output_list, axis=0))
    print("final accuracy", true_len / total_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--eval_freq", default=50, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--load_local", default=False, action="store_true")
    parser.add_argument("--pretrain", default=False, action="store_true")

    parser.add_argument("--image_size", default=300, type=int)
    parser.add_argument("--sort_kinds", default=4, type=int)
    parser.add_argument("--model_detail", default="resnet18", type=str)
    args = parser.parse_args()
    main(args)
