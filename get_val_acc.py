import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from utils import predict_multiple, CIFAR10NP, TRANSFORM, ensure_file_dir

def calculate_acc(dataloader, model, device):
    correct = []
    for imgs, labels in iter(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        pred, _ = predict_multiple(model, imgs)
        correct.append(pred.squeeze(1).eq(labels).cpu())
    correct = torch.cat(correct).numpy()
    return np.mean(correct)

def cal_val_acc(dataset_path, model_name, save_path):
    print(f"===> Calculating {model_name} accuracy for validation sets")
    if os.path.isfile(save_path):
        accuracies = np.load(save_path)
    else:
        train_set = "train_data"
        val_sets = sorted(["val/cifar10-f-32", "val/cifar-10.1-c", "val/cifar-10.1"])

        batch_size = 500
        device = "cuda" 
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

        if not os.path.exists(save_path):
            val_candidates = []
            val_paths = [os.path.join(dataset_path, set_name) for set_name in val_sets]
            for val_path in val_paths:
                if not os.path.exists(val_path):
                    os.makedirs(val_path)
                for file in sorted(os.listdir(val_path)):
                    val_candidates.append(f"{val_path}/{file}")

            accuracies = np.zeros(len(val_candidates))
            for i, candidate in enumerate(tqdm(val_candidates)):
                data_path = f"{candidate}/data.npy"
                label_path = f"{candidate}/labels.npy"

                dataloader = torch.utils.data.DataLoader(
                    dataset=CIFAR10NP(
                        data_path=data_path,
                        label_path=label_path,
                        transform=TRANSFORM,
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                )
                accuracies[i] = calculate_acc(dataloader, model, device)

            accuracies = np.round(accuracies, decimals=6) * 100
            ensure_file_dir(save_path)
            np.save(save_path, accuracies)
            del model
        
    print("===> Successing!")
    return accuracies

if __name__ == "__main__":
    get_val_acc('data/','resnet', 'features/resnet/acc/val_sets.npy')

    
