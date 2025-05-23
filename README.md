
## MLP-Mixer Reproduction

This repository contains our efforts to replicate the fine-tuning and training experiments of the **MLP-Mixer** architecture from the NeurIPS 2021 paper:

> **MLP-Mixer: An all-MLP Architecture for Vision**  
> Tolstikhin *et al.*, NeurIPS 2021  
> https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html

We forked the original code from Google Research’s [vision_transformer](https://github.com/google-research/vision_transformer) and added configuration files to run **9 fine-tuning**  experiments on CIFAR-10 and ImageNet-1K.

---
## Repository Structure
```

├── .github/             ← CI workflows
├── vit\_jax/             ← JAX implementation
│   ├── configs/         ← Experiment configs
│   └── …                ← Requirements, Source code, data loaders, training scripts
├── .gitignore
├── README.md
└── LICENSE

```
---

## Config Files

We provide config files for different experiments. All live under `vit_jax/configs/`.

## Fine-tuning (pretrained → downstream)

| Pretraining Dataset       | Dataset     | Model      | Config file                               |
|:-------------------------:|:-----------:|:----------:|:------------------------------------------|
| ImageNet-1k               | CIFAR-10    | Mixer-B/16 | `mixer_base16_cifar10_imagenet1k.py`                 |
| ImageNet-1k               | CIFAR-10    | Mixer-L/16 | `mixer_large16_cifar10_imagenet1k.py`                |
| ImageNet-21k              | CIFAR-10    | Mixer-B/16 | `mixer_base16_cifar10_imagenet21k.py`                 |
| ImageNet-21k              | CIFAR-10    | Mixer-L/16 | `mixer_large16_cifar10_imagenet21k.py`                |
| ImageNet-21k              | CIFAR-100   | Mixer-B/16 | `mixer_base16_cifar100.py`                |
| ImageNet-21k              | ImageNet-1K | Mixer-B/16 | `mixer_base16_imagenet1k_imagenet21k.py`              |
| ImageNet-21k              | ImageNet-1K | Mixer-L/16 | `mixer_large16_imagenet1k_imagenet21k.py`             |



## Training from scratch (no pretraining)

| Dataset    | Model      | Config file                                           |
|:----------:|:-----------|:------------------------------------------------------|
| CIFAR-10   | Mixer-B/16 | `mixer_base16_cifar10_nopretrain.py`                  |
| CIFAR-10   | Mixer-L/16 | `mixer_large16_cifar10_nopretrain.py`                 |
| ImageNet-1K| Mixer-B/16 | `mixer_base16_imagenet1k-nopretrain.py`               |
| ImageNet-1K| Mixer-L/16 | `mixer_large16_imagenet1k-nopretrain.py`              |

---

## Prerequisites

- **Hardware**: 3× NVIDIA 4090 (or equivalent)  
- **OS**: Linux  
- **Python**: 3.8–3.10  
- **Dependencies**:  

  ```bash
    pip install -r vit_jax/requirements.txt
    pip install tensorflow-datasets jax jaxlib  # for data loading
  ```

---

## Data Preparation

1. **CIFAR-10 & CIFAR-100**

   ```bash
   python3 - <<'EOF'
   import tensorflow_datasets as tfds
   tfds.load('cifar10',  data_dir='$HOME/tensorflow_datasets', download=True)
   tfds.load('cifar100', data_dir='$HOME/tensorflow_datasets', download=True)
   EOF
   ```

2. **ImageNet-1K**

   ```bash
   wget --auth-no-challenge \
     --user=<USERNAME> --password=<PASSWORD> \
     -P /devkit \
     https://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz

   python /home/$USER/tensorflow_datasets/imagenet/unzip.py \
     --data_root=$HOME/tensorflow_datasets/imagenet \
     --devkit_path=/devkit/ILSVRC2012_devkit_t12.tar.gz
   ```

---

## How to Run

Replace `$(pwd)` with your project root if needed.

---

### Fine-tuning on CIFAR-10

```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_b16, category: logs/mixer_b16_cifar10_ft_3gpu
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft/mixer_b16_cifar10_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_cifar10_ft/mixer_b16_cifar10_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```
OR
```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet21k
  CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10-imagenet21k.py" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120 \
  --config.accum_steps=4 \
  > $(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```





```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_l16_cifar10_ft_3gpu
  CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10-imagenet21k.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=120 \
  --config.batch_eval=120 \
  --config.accum_steps=4 \
  > $(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```
OR
```bash
#  Dataset: CIFAR10, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k
  CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10-imagenet21k.py.py" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120 \
  --config.accum_steps=4 \
  > $(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```




```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_b16, category: logs/mixer_b16_cifar10_ft_3gpu_up_imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft/mixer_b16_cifar10_ft_3gpu_up_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet1k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_cifar10_ft/mixer_b16_cifar10_ft_3gpu_up_imagenet/mylog.log 2>&1 & disown
```
OR
```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_b16, category: logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_up_imagenet
L16 cifar10----imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10-imagenet1k.py" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120\
  --config.accum_steps=4\
  > $(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet/mylog.log 2>&1 & disown
```



```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_l16, category: logs/mixer_l16_cifar10_ft_3gpu_up_imagenet
L16 cifar10----imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft/mixer_l16_cifar10_ft_3gpu_up_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10.py" \
  --config.pretrained_dir="gs://mixer_models/imagenet1k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_cifar10_ft/mixer_l16_cifar10_ft_3gpu_up_imagenet/mylog.log 2>&1 & disown
```
OR
```bash
#  Dataset: CIFAR10, Pretrain:imagenet1k, Model: mider_l16, category: logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_up_imagenet
L16 cifar10----imagenet
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_up_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10-imagenet1k.py" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120\
  --config.accum_steps=4\
  > $(pwd)/logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_up_imagenet/mylog.log 2>&1 & disown
```


---

### Fine-tuning on ImageNet-1K

> **Note**: the paper recommends `base_lr = 0.003` for ImageNet-1K fine-tuning; we observed some instability (accuracy dips around 61 %), likely due to randomness in initialization.

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
OR
```bash
#  Dataset: imagenet1k, Pretrain:imagenet21k, Model: mider_b16, category: logs/mixer_imagenet_ft_btsize510/mixer_b16_imagenet_ft_3gpu_imagenet21k
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft_btsize510/mixer_b16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_imagenet1k_imagenet21k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120\
  --config.accum_steps=4\
  > $(pwd)/logs/mixer_imagenet_ft_btsize510/mixer_b16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```



```bash
#  Dataset: imagenet1k, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_l16_imagenet_ft_3gpu_imagenet21k
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_imagenet1k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.pretrained_dir="gs://mixer_models/imagenet21k" \
  --config.batch=510 \
  --config.batch_eval=510\
  --config.accum_steps=5\
  > $(pwd)/logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```
OR
```bash
#  Dataset: imagenet1k, Pretrain:imagenet21k, Model: mider_l16, category: logs/mixer_imagenet_ft_btsize510/mixer_l16_imagenet_ft_3gpu_imagenet21k
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_ft_btsize510/mixer_l16_imagenet_ft_3gpu_imagenet21k" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_imagenet1k_imagenet21k.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.pretrained_dir="$(pwd)/pretrained"\
  --config.batch=120 \
  --config.batch_eval=120\
  --config.accum_steps=4\
  > $(pwd)/logs/mixer_imagenet_ft_btsize510/mixer_l16_imagenet_ft_3gpu_imagenet21k/mylog.log 2>&1 & disown
```

---

### Training from Scratch (CIFAR-10)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_b16_cifar10_training" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_cifar10_nopretrain.py" \
  --config.batch=510 \
  --config.batch_eval=510
  > $(pwd)/logs/mixer_b16_cifar10_training/mylog.log 2>&1 & disowns
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_l16_cifar10_training" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_cifar10_nopretrain.py" \
  --config.batch=510 \
  --config.batch_eval=510
  > $(pwd)/logs/mixer_l16_cifar10_training/mylog.log 2>&1 & disowns
```

---

### Training from Scratch (ImageNet-1K)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_training/mixer_b16_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_base16_imagenet1k-nopretrain.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.batch=510 \
  --config.batch_eval=510\
  > $(pwd)/logs/mixer_imagenet_training/mixer_b16_imagenet/mylog.log 2>&1 & disown
```


```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m vit_jax.main \
  --workdir="$(pwd)/logs/mixer_imagenet_training/mixer_l16_imagenet" \
  --config="$(pwd)/vit_jax/configs/mixer_large16_imagenet1k-nopretrain.py" \
  --config.dataset="/home/dddddddd/tensorflow_datasets/imagenet" \
  --config.batch=510 \
  --config.batch_eval=510\
  > $(pwd)/logs/mixer_imagenet_training/mixer_l16_imagenet/mylog.log 2>&1 & disown
```

---

## Expected Results of CIFAR10 from the Paper
batchsize=512
total_steps=1000
upstream     | model      | dataset | top-1 accuracy |
:----------- | :--------- | :------ | -------------: |
ImageNet     | Mixer-B/16 | cifar10 | 96.72%         |
ImageNet     | Mixer-L/16 | cifar10 | 96.59%         |
ImageNet-21k | Mixer-B/16 | cifar10 | 96.82%         |
ImageNet-21k | Mixer-L/16 | cifar10 | 98.34%         |


## Our Results
batchsize=510*5=2550
total_steps=10000
upstream     | model      | dataset | top-1 accuracy | 
:----------- | :--------- | :------ | -------------: | 
ImageNet-21k | Mixer-L/16 | cifar10 | 98.29%         | logs/mixer_cifar10_ft/mixer_l16_cifar10_ft_3gpu_imagenet21k

### We did the upper experiment because the 1:1 comparason below gets lower accuracy than that in the paper. We find that longer total_steps may get limited higher test accuracy.

batchsize=120*4=480
total_steps=1000
upstream     | model      | dataset | top-1 accuracy | 
:----------- | :--------- | :------ | -------------: | 
ImageNet     | Mixer-B/16 | cifar10 | 94.78%         |√logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet
ImageNet     | Mixer-L/16 | cifar10 | 95.79%         |√logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_up_imagenet
ImageNet-21k | Mixer-B/16 | cifar10 | 96.44%         |√logs/mixer_cifar10_ft_btsize510/mixer_b16_cifar10_ft_3gpu_imagenet21k
ImageNet-21k | Mixer-L/16 | cifar10 | 98.18%         |√logs/mixer_cifar10_ft_btsize510/mixer_l16_cifar10_ft_3gpu_imagenet21k

---

## Expected Results of ImageNet-1k from the Paper

batchsize=512
total_steps=125000
upstream     | model      | dataset     | top-1 accuracy | 
:----------- | :--------- | :---------- | -------------: | 
ImageNet-21k | Mixer-B/16 | ImageNet-1k | 80.64%         |
ImageNet-21k | Mixer-L/16 | ImageNet-1k | 82.89%         |


## Our Results

batchsize=510*5=2550
total_steps=125000
upstream     | model      | dataset     | top-1 accuracy | 
:----------- | :--------- | :---------- | -------------: | 
ImageNet-21k | Mixer-B/16 | ImageNet-1k | 78.47%         | logs/mixer_imagenet_ft/mixer_b16_imagenet_ft_3gpu_imagenet21k


batchsize=510*5=2550
total_steps=35000
upstream     | model      | dataset     | top-1 accuracy | 
:----------- | :--------- | :---------- | -------------: | 
ImageNet-21k | Mixer-L/16 | ImageNet-1k | 81.84%         | logs/mixer_imagenet_ft/mixer_l16_imagenet_ft_3gpu_imagenet21k

### We did the upper 2 experiments because the 1:1 comparason below gets lower accuracy than that in the paper. We find that longer total_steps may get limited higher test accuracy, but lower than that in the paper.

batchsize=120*4=480
total_steps=50000
upstream     | model      | dataset     | top-1 accuracy | 
:----------- | :--------- | :---------- | -------------: | 
ImageNet-21k | Mixer-B/16 | ImageNet-1k | 77.57%         | logs/mixer_imagenet_ft/mixer_b16_imagenet_ft_3gpu_imagenet21k
ImageNet-21k | Mixer-L/16 | ImageNet-1k | 81.44%         | logs/mixer_imagenet_ft/mixer_b16_imagenet_ft_3gpu_imagenet21k

---



## License

This code derives from Google Research’s [vision\_transformer](https://github.com/google-research/vision_transformer) (Apache 2.0). See [LICENSE](LICENSE) for details.
