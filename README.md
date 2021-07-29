# ICLib
Image Classification Library Using Deep Learning Methods. (PyTorch Implementation)



## Requirements

1.   Create conda environment

```bash
conda create -n torch python=3.7
source activate torch
conda install numpy=1.19.1
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch 
conda install tqdm pymongo
pip install sacred
```

2.   Setup [MongoDB and Omniboard](doc/MongoDB+Omniboard.md) used for recording experiments by [Sacred](https://github.com/IDSIA/sacred).



## Self-Supervised Learning

The part of self-supervised learning is an PyTorch implement of SimCLR.



### 1. Usage

*   Pretrain

```bash
python main.py pretrain with pretrain_cifar10
```

*   Finetune

```bash
python main.py train_then_eval with train_then_eval_cifar10 ckpt_id=1
```



### 2. Results

| Model                                | Resource | CIFAR-10 |
| ------------------------------------ | -------- | -------- |
| SimCLR (ResNet18)                    | 5GB      | 92.43    |
| SimCLR (ResNet50)                    | 18GB     | 93.25    |
| SimCLR (ResNet50), ImageNet-Pretrain |          | 93.08    |



## Acknowledgement

Thanks to [google-research/simclr](https://github.com/google-research/simclr) and [sthalles/SimCLR](https://github.com/sthalles/SimCLR) .

