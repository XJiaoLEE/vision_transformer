We try to replicate the experiment results of mixer_mlp fine tuning and training from this paper.

We designed the following config files in order to run the 6 fine tuning and 2 training experiments.

- Fine tuning configs:
  - /vit_jax/configs/mixer_base16_cifar10.py
  - /vit_jax/configs/mixer_large16_cifar10.py
  - /vit_jax/configs/mixer_base16_cifar100.py
  - /vit_jax/configs/mixer_base16_imagenet1k.py
  - /vit_jax/configs/mixer_large16_imagenet1k.py

- Training configs:
  - /vit_jax/configs/mixer_base16_cifar10_nopretrain.py
  - /vit_jax/configs/mixer_large16_cifar10_nopretrain.py
  - /vit_jax/configs/mixer_base16_imagenet1k-nopretrain.py
  - /vit_jax/configs/mixer_large16_imagenet1k-nopretrain.py




Steps to replicate experiments:

- 数据集下载和预处理：
1. CIFAR10 & CIFAR100

  提前下载cifar10和cifar100
  python3 - <<'EOF'                                     
  import tensorflow_datasets as tfds                                                                            
  # 第一次运行会下载并准备好cifar10             
  tfds.load('cifar10', data_dir='$HOME/tensorflow_datasets', download=True)
  EOF

  python3 - <<'EOF'                                     
  import tensorflow_datasets as tfds                                                                            
  # 第一次运行会下载并准备好cifar100             
  tfds.load('cifar100', data_dir='$HOME/tensorflow_datasets', download=True)
  EOF

2. Imagenet1K
wget --auth-no-challenge \
     --user=xli886  --password=LXJnotjiao123! \
     -P /devkit \
     https://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz

然后用/home/dddddddd/tensorflow_datasets/imagenet/unzip.py来处理数据


3. 硬件配置：
4090GPU*3



4. 实验
(1) CIFAR10 fine tuning *4，对比论文源码README文件中的293-304行的实验结果粘贴如下：
### Expected Mixer results

We ran the fine-tuning code on Google Cloud machine with four V100 GPUs with the
default adaption parameters from this repository. Here are the results:

upstream     | model      | dataset | accuracy | wall_clock_time | link
:----------- | :--------- | :------ | -------: | :-------------- | :---
ImageNet     | Mixer-B/16 | cifar10 | 96.72%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/j9zCYt9yQVm93nqnsDZayA/)
ImageNet     | Mixer-L/16 | cifar10 | 96.59%   | 3.0h            | [tensorboard.dev](https://tensorboard.dev/experiment/Q4feeErzRGGop5XzAvYj2g/)
ImageNet-21k | Mixer-B/16 | cifar10 | 96.82%   | 9.6h            | [tensorboard.dev](https://tensorboard.dev/experiment/mvP4McV2SEGFeIww20ie5Q/)
ImageNet-21k | Mixer-L/16 | cifar10 | 98.34%   | 10.0h           | [tensorboard.dev](https://tensorboard.dev/experiment/dolAJyQYTYmudytjalF6Jg/)



```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_b16, category: logs/mixer_b16_cifar10_ft_3gpu
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_b16_cifar10_ft_3gpu" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5
```



```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_l16_cifar10_ft_3gpu
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_l16_cifar10_ft_3gpu" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_l16_cifar10_ft_3gpu/mylog.log 2>&1 & disown
```


```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_b16, category: logs/mixer_b16_cifar10_ft_3gpu_up_imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_b16_cifar10_ft_3gpu_up_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet1k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_b16_cifar10_ft_3gpu_up_imagenet/mylog.log 2>&1 & disown
```


```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_l16, category: logs/mixer_l16_cifar10_ft_3gpu_up_imagenet
L16 cifar10----imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_l16_cifar10_ft_3gpu_up_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet1k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_l16_cifar10_ft_3gpu_up_imagenet/mylog.log 2>&1 & disown
```




(2) imagenet1k fine tuning *4
------------------------------------------------------------------------------------------------------
```bash
#  Dataset: imagenet1k, Pretrain:imagenet21k, Model: mider_b16, category: logs/mixer_b16_imagenet_ft_3gpu_imagenet21k
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft/mixer_b16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_imagenet1k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_imagenet_ft/mixer_b16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```
问题：按照论文，在微调Imagenet1k时应该设置config.base_lr = 0.003，但是在第一次训练时，模型在达到61% testing_accuracy后开始下降。第三次训练才逐渐提升。存在随机性影响训练效果。




2. 21k预训练
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_imagenet1k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown




纯训练------------------
2. 1k预训练
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_imagenet1k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.batch=510 \
  --config.batch_eval=510\
  > $(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown














实验4：MLP-Mixer-S/16快速训练（CIFAR-10，3GPU）
bash

# 小模型从头训练，适合CIFAR-10
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir=$(pwd)/logs/mixer_s16_cifar10_3gpu \
  --config=$(pwd)/vit_jax/configs/mixer.py:s16,cifar10 \
  --config.batch=510 \                
  --config.pp.crop=32 \
  --config.epochs=100 \
  --config.base_lr=0.01 \
  --config.batch_eval=510

    预期结果：100 epoch后Top-1 ~90%。


CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir=$(pwd)/logs/mixer_s16_cifar10_3gpu \
  --config=$(pwd)/vit_jax/configs/mixer_s16_cifar10.py \
  --config.batch=510 \    
  --config.base_lr=0.01 \
  --config.batch_eval=510



  CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir=$(pwd)/logs/mixer_s16_cifar10_3gpu \
  --config=$(pwd)/vit_jax/configs/mixer_s16_cifar10.py \    
  --config.base_lr=0.01 











