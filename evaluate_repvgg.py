import argparse
import os
import sys
from tqdm import tqdm
import numpy as np

sys.path.append(".")

from scipy import linalg
import scipy.stats
from sklearn.metrics import mean_squared_error

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import functional as F

from utils import predict_multiple, store_ans, ensure_file_dir, mse_loss, cal_param_size, CIFAR10NP, TRANSFORM, CIFAR10NP_TEST
from atc import get_atc_threshold
from fid import get_fid_feature, calculate_frechet_distance
from model import Mlp, Trans
from get_val_acc import cal_val_acc

from IPython import embed


parser = argparse.ArgumentParser(description="AutoEval baselines - Entropy")
parser.add_argument("--dataset_path", default='data/', type=str, help="path containing all datasets (training and validation)",)
parser.add_argument("--test", default=True, type=bool, help="evaluate on the test set")
parser.add_argument('-l',"--load_model", default='', type=str, help="load the checkpoint for verify the result directly")
parser.add_argument('-e', "--ensemble", default='', type=str, help="two machine learning model ensemble. e.g: -e model_cnn.pth+model_vit.pth")
parser.add_argument('-m', "--model", default='Mlp', type=str, help="two model structures: Mlp or Trans")


use_metric = ['entropy', 'atc', 'fid', 'stat']
use_fid = True if 'fid' in use_metric else False 

def calculate_metric(dataloader, model, device, atc_thre, fid_mean, fid_cov, candidate, stage, save_root, metrics=['entropy', 'maxconf', 'atc', 'fid', 'stat']):
    entropy_scores = []
    maxconf_scores = []
    atc_scores = []
    fid_scores = []

    real_pred = []
    real_feat = []

    img_mean = []
    img_std = []
    img_iep = []
    img_lp = []

    corr = []
    test = True if stage=='test' else False

    if not os.path.exists(f'{save_root}/predict/{stage}'):
        os.makedirs(f'{save_root}/predict/{stage}') 
    
    if not os.path.exists(f'features/img_statics/{stage}'):
        os.makedirs(f'features/img_statics/{stage}')
     
    logits_path = '{}/predict/{}/logits_{}'.format(save_root, stage, candidate)
    feat_path = '{}/predict/{}/feat_{}'.format(save_root, stage, candidate)
    static_path = 'features/img_statics/{}/{}'.format(stage, candidate)
    fid_path = '{}/predict/{}/fid_{}'.format(save_root, stage, candidate)

    if os.path.isfile(static_path) and 'stat' in metrics :
        ifs = np.load(static_path)
        img_mean = ifs[:,0].tolist()
        img_std = ifs[:,1].tolist()
        img_iep = ifs[:,2].tolist()
        img_lp = ifs[:,3].tolist()

    if os.path.isfile(fid_path) and 'fid' in metrics:
        fid_scores = np.load(fid_path).tolist()

    if os.path.isfile(logits_path) and os.path.isfile(feat_path):
        real_pred = np.load(logits_path)
        if not os.path.isfile(fid_path):
            feat = np.load(feat_path)
        ep = scipy.stats.entropy(real_pred, axis=1)
        if 'entropy' in metrics:
            entropy_scores = ep.tolist()
        if 'maxconf' in metrics:
            maxconf_scores = np.max(real_pred, axis=1).tolist()
        if 'atc' in metrics:
            atc_scores = (ep/atc_thre).tolist()
        if not test:
            labels = dataloader.dataset.labels
            corr = ((np.argmax(real_pred, axis=1) == labels) + 0.0).tolist()
    else:
        def forward_hook_fn(module, input, output):
            real_feat.append(output.squeeze().detach().cpu())
        
        hk = list(model.children())[-2].register_forward_hook(forward_hook_fn)

        if 'stat' in metrics and len(img_mean) == 0:
            imgs = dataloader.dataset.imgs.reshape(len(dataloader.dataset),-1)
            img_mean = imgs.reshape(len(dataloader.dataset),-1).mean(axis=-1).tolist()
            img_std = imgs.reshape(len(dataloader.dataset),-1).std(axis=-1).tolist()
            imgs = imgs / 256.
            img_iep = (-imgs*np.log(imgs+1e-6)).mean(axis=-1).tolist()
            
        for batch in iter(dataloader):
            try:
                imgs, labels, lps = batch
                if not os.path.isfile(static_path):
                    img_lp.extend(lps.numpy().reshape(-1).tolist())
            except:
                imgs, labels  = batch
            imgs, labels = imgs.to(device), labels.to(device)        
            
            model.eval()
            with torch.no_grad():
                prob = model(imgs)
                prob = torch.nn.functional.softmax(prob, dim=1).cpu().numpy()
            
            real_pred.append(prob)
            if not test:
                corr.extend(((np.argmax(prob, axis=-1)==labels.cpu().numpy())+0.0).tolist())
            ep = scipy.stats.entropy(prob, axis=1)

            if 'entropy' in metrics:
                entropy_scores.extend(ep.tolist())
            if 'maxconf' in metrics:
                maxconf_scores.extend(np.max(prob, axis=1).tolist())
            if 'atc' in metrics:
                atc_scores.extend((ep/atc_thre).tolist())
                
        hk.remove()
        real_pred = np.concatenate(real_pred, axis=0)
        np.save(logits_path, real_pred)

        feat = torch.cat(real_feat).numpy()
        np.save(feat_path, feat)
        real_feat = []

        if not os.path.isfile(static_path) and 'stat' in metrics:
            all_stat = np.stack([img_mean, img_std, img_iep, img_lp], axis=1) 
            np.save(static_path, all_stat)

    if 'fid' in metrics and len(fid_scores)==0:
        mu = np.mean(feat, axis=0)
        sigma = np.cov(feat, rowvar=False)
        fd = calculate_frechet_distance(fid_mean, fid_cov, mu, sigma)
        fid_scores = np.array([fd]*feat.shape[0])
        np.save(fid_path, fid_scores)

    res = []
    for s in [entropy_scores, maxconf_scores, atc_scores, img_std, img_iep, img_lp, fid_scores, corr]:
        if len(s) > 0:
            res.append(s)
    res = np.stack(res, axis=-1)
    return res

global fid_norm
def normalization(data, training=True):
    global fid_norm

    if not use_fid:
        _range = np.max(data, axis=0) - np.min(data, axis=0)
        return (data - np.min(data, axis=0)) / _range

    if training:
        data_fid = data[:,-1]
        fid_min = data_fid.min()
        fid_range = np.max(data_fid) - np.min(data_fid)
        fid_norm = [fid_min, fid_range]

        _range = np.max(data, axis=0) - np.min(data, axis=0)
        return (data - np.min(data, axis=0)) / _range
    else:
        data_fid = data[:,-1]
        data = data[:,:-1]

        _range = np.max(data, axis=0) - np.min(data, axis=0)
        data = (data - np.min(data, axis=0)) / _range

        fid_min, fid_range = fid_norm
        data_fid = (data_fid - fid_min) / fid_range
        data_fid = np.maximum(np.minimum(data_fid, 1.), 0.0).reshape(-1,1)
        data = np.concatenate([data, data_fid], axis=-1)
        return data

def judge_read_dataset(feature_save_root, stage, nums):
    need_read_dataset = True
    if os.path.isdir(f'{feature_save_root}/predict/{stage}'):
        all_files = os.listdir(f'{feature_save_root}/predict/{stage}')
        logits_files = list(filter(lambda x: x.startswith('logits'), all_files))
        feat_files = list(filter(lambda x: x.startswith('feat'), all_files))
        fid_files = list(filter(lambda x: x.startswith('fid'), all_files))
        if len(logits_files) == len(feat_files) == len(fid_files) == nums:
            need_read_dataset = False
    return need_read_dataset

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_path = args.dataset_path
    model_name = 'repvgg'
    train_set = "train_data"
    val_sets = sorted(["val/cifar10-f-32", "val/cifar-10.1-c", "val/cifar-10.1"])

    feature_save_root = f"features/{model_name}"

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

    assert args.model in ['Mlp', 'Trans'], 'only support two model structures: Mlp and Trans'

    atc_thre = get_atc_threshold(model_name, dataset_path, f'{feature_save_root}/atc_thre.npy')
    fid_mean, fid_cov = get_fid_feature(model_name, dataset_path,  f'{feature_save_root}/fid_mean.npy', f'{feature_save_root}/fid_cov.npy')

    train_path = os.path.join(dataset_path, train_set)
    train_candidates = []
    for file in sorted(os.listdir(train_path)):
        if file.endswith(".npy") and file.startswith("new_data"):
            train_candidates.append(file)

    metrics = []
    print("===> Calculating all metric for train sets")

    need_read_dataset = judge_read_dataset(feature_save_root, 'train', 1000)
    for i, candidate in enumerate(tqdm(train_candidates)):
        data_path = os.path.join(train_path, candidate)
        label_path = os.path.join(train_path, 'labels.npy')
        dataloader = torch.utils.data.DataLoader(
            dataset=CIFAR10NP(
                data_path=data_path,
                label_path=label_path,
                transform=TRANSFORM,
                st=need_read_dataset,
                need_read=need_read_dataset
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        
        metrics.append(calculate_metric(dataloader, model, device, atc_thre, fid_mean, fid_cov, candidate, 'train', feature_save_root, metrics=use_metric))
    train = np.concatenate(metrics)
    print(f"===> Successing! ")

    print(f"===> Calculating all metric for validation sets")
    val_candidates = []
    val_paths = [os.path.join(dataset_path, set_name) for set_name in val_sets]
    for val_path in val_paths:
        for file in sorted(os.listdir(val_path)):
            val_candidates.append(os.path.join(val_path, file))

    need_read_dataset = judge_read_dataset(feature_save_root, 'val', 40)
    val_metrics = []
    for i, candidate in enumerate(tqdm(val_candidates)):
        data_path = f"{candidate}/data.npy"
        label_path = f"{candidate}/labels.npy"

        dataloader = torch.utils.data.DataLoader(
            dataset=CIFAR10NP(
                data_path=data_path,
                label_path=label_path,
                transform=TRANSFORM,
                st=need_read_dataset,
                need_read=need_read_dataset
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        val_metrics.append(calculate_metric(dataloader, model, device, atc_thre, fid_mean, fid_cov, f"{i}.npy", 'val', feature_save_root, metrics=use_metric))
    print(f"===> Successing! ")

    train_y = train[:,-1]
    train_x = normalization(train[:,:-1], training=True)

    dp = 0.0 if args.model=='Mlp' else 0.1
    method = eval(args.model)(train_x.shape[1], drop=dp).cuda()
    ms = cal_param_size(method) / 1e3
    print(f"===> Train a tiny model({ms} K) for evaluating accuracy with model: {model_name}")

    if args.ensemble != '':
        assert '+' in args.ensemble, 'when ensemble, input muse be model_path1+model_path2'
        model1_path, model2_path = args.ensemble.split('+')
        m2 = 'Trans' if args.model=='Mlp' else 'Mlp'
        method2 = eval(m2)(train_x.shape[1]).cuda()
        try:
            method.load_state_dict(torch.load(model1_path))
            method2.load_state_dict(torch.load(model2_path))
        except:
            method.load_state_dict(torch.load(model2_path))
            method2.load_state_dict(torch.load(model1_path))
        method2.eval()
    elif args.load_model != '' and os.path.isfile(args.load_model):
        print(f"===> Loading the checkpoint: {args.load_model}")
        method.load_state_dict(torch.load(args.load_model))
    else:
        max_iter = 30000
        base_lr = 0.1 if args.model=='Mlp' else 0.05
        params_list = nn.ModuleList([])
        params_list.append(method)
        decay = []
        no_decay = []

        for name, param in method.named_parameters():
            if ('bn' in name or 'bias' in name):
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{'params': decay}, {'params': no_decay, 'weight_decay': 0.0}]

        optimizer = torch.optim.SGD(per_param_args,lr=base_lr,momentum=0.9, weight_decay=5e-4)
        
        def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
            cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            return cur_lr

        train_x = torch.tensor(train_x).cuda().float()
        train_y = torch.tensor(train_y).cuda().float()

        split = train_x.shape[0] // 1000

        for iters in range(max_iter):
            ind = iters % 1000
            x = train_x[ind*split:(ind+1)*split]
            y = train_y[ind*split:(ind+1)*split]
            method.train()
            optimizer.zero_grad()
            pred = method(x)
            loss = mse_loss(pred, y)
            cur_lr = adjust_lr(optimizer, base_lr, iters, max_iter)
            loss.backward()
            optimizer.step()
            acc = (pred>0.5).sum() / y.shape[0]
            if iters % 100 == 0:
                print('ITER: {}, LR: {:.4f}, loss: {:.4}, Pred_ACC: {}, Real_ACC: {}'.format(iters, cur_lr, loss, acc.item(), y.mean()))
        
        save_model_path = 'result/{}/model.pth'.format(model_name)
        ensure_file_dir(save_model_path)
        torch.save(method.state_dict(),save_model_path)
    print(f"===> Successing! ")

    val_res = cal_val_acc(dataset_path, model_name, f'features/{model_name}/acc/val_sets.npy')
    print("===> Running on val sets")
    val_pred = {}
    for val_s in val_metrics:
        val_y = val_s[:,-1]
        val_x = normalization(val_s[:,:-1], training=False)
        method.eval()
        val_x = torch.tensor(val_x).cuda().float()

        with torch.no_grad():
            val_y_hat = method(val_x).cpu().numpy().reshape(-1)
            if args.ensemble != '':
                val_y_hat2 = method2(val_x).cpu().numpy().reshape(-1)
                val_y_hat = (val_y_hat + val_y_hat2) / 2.0

        for thre in range(45,60,5):
            thre = thre / 100.0
            if thre not in val_pred:
                val_pred[thre] = []
            val_pred[thre].append((val_y_hat > thre).sum() / val_y_hat.shape[0] * 100)

    best_thre = 0.5
    best_loss = 10000000
    for k,v in val_pred.items():
        rmse_loss = mean_squared_error(y_true=val_res, y_pred=v, squared=False)
        print(f"The RMSE on validation set is @ thre {k}: {rmse_loss}")
        if rmse_loss < best_loss:
            best_thre =  k
            best_loss = rmse_loss
    print('*'*20)
    print('NOTE: The RMSE on val set should be in [4.2, 4.4] when threshold is set to 0.45. If False, please try again!')
    print('*'*20)
    print("===> Successing! ")

    if args.test:
        print("===> Running on TEST sets")
        test_root = os.path.join(dataset_path, 'test_data')
        test_candidates = sorted(os.listdir(test_root))
        test_entscores = []
        for i, candidate in enumerate(tqdm(test_candidates)):
            data_path = os.path.join(test_root, candidate)

            dataloader = torch.utils.data.DataLoader(
                dataset=CIFAR10NP_TEST(
                    data_path=data_path,
                    transform=TRANSFORM,
                    st=True
                ),
                batch_size=batch_size,
                shuffle=False,
            )
            metric= calculate_metric(dataloader, model, device, atc_thre, fid_mean, fid_cov, candidate, 'test', feature_save_root, metrics=use_metric)
            test_entscores.append(metric)

        accuracy = []
        for val_s in test_entscores:
            val_x = normalization(val_s, training=False)
            method.eval()
            
            val_x = torch.tensor(val_x).cuda().float()

            with torch.no_grad():
                val_y_hat = method(val_x).cpu().numpy().reshape(-1)
                if args.ensemble != '':
                    val_y_hat2 = method2(val_x).cpu().numpy().reshape(-1)
                    val_y_hat = (val_y_hat + val_y_hat2) / 2.0

            accuracy.append( (val_y_hat > best_thre).sum() / val_y_hat.shape[0] )

        result_path = f'result/{model_name}/predict.txt'
        ensure_file_dir(result_path)
        store_ans(accuracy,result_path)
        print(f'The result file path is: {result_path}')
        print('Successing! ')
