# Scoring Your Prediction !

This is the official implementation of the 1st place for the [DATACV](https://sites.google.com/view/vdu-cvpr23/competition?authuser=0) competition (CVPR 2023 Workshop).

We just train a tiny model (about 6 KB) to score another model predictions (ResNet or RepVGG) ! We can obtain the exciting results when estimating accuracy on label-free test set.

### Prepare

- Before running the code, you should install torch(1.13.0) and torchvision(0.14.0).
- Download the origin train/val/test data from [link](https://github.com/xingjianleng/autoeval_baselines) into DATASET.
- Create a symbolic link in the ROOT

```bash
ln -s DATASET data
```
data folder structure :
```
data
	- cifar10
	- test_data
	- train_data
	- val
```

### Usage

Note that the codebase only support running on the single GPU. You only need run a line code for reproduction.

For ResNet: 
```bash
python3 evaluate_resnet.py
```

For RepVGG: 
```bash
python3 evaluate_repvgg.py
```

The RMSE results of ResNet/RepVGG on the validation sets are as follows:

model  | RMSE
------------- | -------------
ResNet  | 3.15
RepVGG  | 4.3

We also provide the checkpoints to re-implement the result we submitted on the leadboard.

For ResNet, 
```bash
python3 evaluate_resnet.py -l checkpoint/resnet/model.pth
```
We must emphasize here that **this checkpoint is not the model that produces the final result of ResNet**. Because of our oversight,  the model has not been saved. This checkpoint is our re-implementation, and the performance of the two models on validation sets is comparable (RMSE on validation sets is about 3.15).

For RepVGG:
```bash
python3 evaluate_repvgg.py -e checkpoint/repvgg/model_cnn.pth+checkpoint/repvgg/model_vit.pth
```
It can produce the same prediction with the best result on leadboard ( for RepVGG). We adopt the ensemble method of two models to obtain the RepVGG prediction due to difficulty

Note that our codebase saves some intermediate results, and thus the speed will be slightly slower when first time running, but it will be very fast when running again.