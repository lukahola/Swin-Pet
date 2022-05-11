<<<<<<< HEAD
# Pet-classification

By 张天佑 SY2117325

## Introduction

研一下学期**图像处理分析与识别**课程大作业，基于*Swin-Transformer*实现猫狗品种分类。

### Swin-Transformer简介

>from [Swin](https://github.com/microsoft/Swin-Transformer)

**Swin Transformer** (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a
general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is
computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

Swin Transformer achieves strong performance on COCO object detection (`58.7 box AP` and `51.1 mask AP` on test-dev) and
ADE20K semantic segmentation (`53.5 mIoU` on val), surpassing previous models by a large margin.

![teaser](E:/Projects/Git/Pet-classification/figures/teaser.png)

### 本项目简介

搜集建立了包含39种不同宠物猫狗的图像数据集，每个种类包含200张图片，总计7900张图片，以imagenet1k的数据格式保存。训练结果acc@1达到96.7%，acc@5达到99.6%。

|  name  |  pretrain   | resolution | acc@1  | acc@5  | model |
| :----: | :---------: | :--------: | :----: | :----: | :---: |
| Swin-B | ImageNet-1K |  224*224   | 96.667 | 99.615 |       |

## Getting started

**十分建议根据原repo提示安装（[get_started.md](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md)），但有部分安装操作存在问题，可以参照此repo进行安装。**

### Install

- Clone this repo:

```bash
git clone https://github.com/lukahola/Pet-classification.git
cd Pet-classification
```

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.7 -y
conda activate pet
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.8.1` and `torchvision==0.9.1` with `CUDA==10.1`(注意版本与原repo不同，实测原repo `PyTorch==1.7.1` and `torchvision==0.8.2`不行):

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

如果出现问题，可以尝试：

```bash
python setup.py install
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
```

### Data preparation

本项目数据集格式基于imagenet1k，结构如下所示：

  ```bash
$ tree data
imagenet
├── train_map.txt
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
├──val_map.txt
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
  ```

  `train_map.txt`和`val_map.txt`的内容格式为

  ```txt
class1/img1.jpeg 1
class1/img2.jpeg 1
...
class2/img45.jpeg 2
...
  ```

  >其余尽可参照原repo。
=======
# Swin-Pet
>>>>>>> 3a2ccc7ce18a52aa519d1a197abf9fd8c73314dd
