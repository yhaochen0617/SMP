
import os
import sys

import numpy as np

import torch
import torch.utils.data
import torchvision.datasets
from tqdm import tqdm
import scipy.stats

from utils import CIFAR10NP, TRANSFORM, predict_multiple, ensure_file_dir



def calculate_threshold(acc, atcs):
    sorted_atcs = np.sort(atcs)
    lower_tail_num = int(np.ceil(acc * len(atcs)))
    return sorted_atcs[lower_tail_num]


def calculate_atcs(dataloader, model, device):
    correct, atcs = [], []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, prob = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
        atcs.extend((scipy.stats.entropy(prob, axis=1)).tolist())
    correct = torch.cat(correct).numpy()
    return np.mean(correct), np.array(atcs)


def get_atc_threshold(model_name, dataset_path, save_path):
    print("===> Calculating the threshold for ATC")
    if os.path.isfile(save_path):
        threshold = np.load(save_path)
    else:
        train_set = "train_data"
        val_sets = sorted(["val/cifar10-f-32", "val/cifar-10.1-c", "val/cifar-10.1"])
        
        batch_size = 500
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == "resnet":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
            )
        elif model_name == "repvgg":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True
            )
        else:
            raise ValueError("Unexpected model_name")
        model.to(device)
        model.eval()

        
        cifar_testloader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_path, 'cifar10'),
                train=False,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        threshold = calculate_threshold(*calculate_atcs(cifar_testloader, model, device))
        ensure_file_dir(save_path)
        np.save(save_path, np.array(threshold))
        del model
    print('===> Successing!')
    return threshold

if __name__ == "__main__":
    print(get_atc_threshold('resnet', 'data', 'features/resnet/atc_thre.npy'))
    
