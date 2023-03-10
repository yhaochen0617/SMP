"""
The Frechet Distance calculation functions utilized in this analysis were sourced from a publicly available repository
https://github.com/Simon4Yan/Meta-set
"""
import argparse
import os
import sys

import numpy as np
from scipy import linalg

import torch
import torch.nn
import torch.utils.data
import torchvision.datasets
from tqdm import tqdm

from utils import CIFAR10NP, TRANSFORM, ensure_file_dir


def get_activations(dataloader, model, dims, device):
    # Calculates the activations of final feature vector for all images
    batch_size = dataloader.batch_size
    n_used_imgs = len(dataloader.dataset)

    pred_arr = np.empty((n_used_imgs, dims))

    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            start = i * batch_size
            end = start + batch_size
            imgs = imgs.to(device)
            pred = model(imgs)
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(imgs.shape[0], -1)
    return pred_arr


def calculate_activation_statistics(dataloader, model, dims, device):
    # Calculation of the statistics used by the FD.
    act = get_activations(dataloader, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Numpy implementation of the Frechet Distance.
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates" % eps
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_fid_feature(model_name, dataset_path, save_path_mean, sava_path_cov):

    print("===> Calculating the FID of original dataset")
    if os.path.isfile(save_path_mean) and os.path.isfile(sava_path_cov):
        fid_mean = np.load(save_path_mean)
        fid_cov = np.load(sava_path_cov)
    else:
        train_set = "train_data"
        val_sets = sorted(["val/cifar10-f-32", "val/cifar-10.1-c", "val/cifar-10.1"])

        batch_size = 500
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == "resnet":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
            )
            model_feat = torch.nn.Sequential(
                *list(model.children())[:-1], torch.nn.Flatten()
            )
            dims = 64
        elif model_name == "repvgg":
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a0", pretrained=True
            )
            model_feat = torch.nn.Sequential(
                *list(model.children())[:-1], torch.nn.Flatten()
            )
            dims = 1280
        else:
            raise ValueError("Unexpected model_name")
        model_feat.to(device)
        model_feat.eval()

        # Use original CIFAR10 training data to calculate reference FID
        cifar_testloader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.CIFAR10(
                root=os.path.join(dataset_path, 'cifar10'),
                train=True,
                transform=TRANSFORM,
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        fid_mean, fid_cov, _ = calculate_activation_statistics(
            cifar_testloader, model_feat, dims, device
        )
        ensure_file_dir(save_path_mean)
        ensure_file_dir(sava_path_cov)
        np.save(save_path_mean, fid_mean)
        np.save(sava_path_cov, fid_cov)
        
        del model_feat
        del model
    print("===> Successing! ")
    return fid_mean, fid_cov

if __name__ == '__main__':
    print(get_fid_feature('resnet', './data', 'features/resnet/fid_mean.npy', 'features/resnet/fid_cov.npy'))


    