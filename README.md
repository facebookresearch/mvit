# [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526)

Official PyTorch implementation of **MViTv2**, from the following paper:

[MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526). CVPR 2022.\
Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer*

---

MViT is a multiscale transformer which serves as a general vision backbone for different visual recognition tasks:

> **Image Classification**: Included in this repo.

> **Object Detection and Instance Segmentation**: See [MViTv2 in Detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/MViTv2).

> **Video Action Recognition and Detection**: See [MViTv2 in PySlowFast](https://github.com/facebookresearch/SlowFast/tree/main/projects/mvitv2).


# Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | 1k model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| MViTv2-T | 224x224 | 82.3 | 24M | 4.7G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_T_in1k.pyth) |
| MViTv2-S | 224x224 | 83.6 | 35M | 7.0G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_S_in1k.pyth) |
| MViTv2-B | 224x224 | 84.4 | 52M | 10.2G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in1k.pyth) |
| MViTv2-L | 224x224 | 85.3 | 218M | 42.1G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in1k.pyth) |

### ImageNet-21K trained models

| name | resolution |acc@1 | #params | FLOPs | 21k model | 1k model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| MViTv2-B | 224x224 | - | 52M | 10.2G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_B_in21k.pyth) | - |
| MViTv2-L | 224x224 | 87.5 | 218M | 42.1G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_L_in21k.pyth) | - |
| MViTv2-H | 224x224 | 88.0 | 667M | 120.6G | [model](https://dl.fbaipublicfiles.com/mvit/mvitv2_models/MViTv2_H_in21k.pyth) | - |

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Training

Here we can train a standard MViTv2 model from scratch by:
```
python tools/main.py \
  --cfg configs/MViTv2_T.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 256 \
```

## Evaluation

To evaluate a pretrained MViT model:
```
python tools/main.py \
  --cfg configs/test/MViTv2_T_test.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 8 \
  TEST.BATCH_SIZE 256 \
```


## Acknowledgement
This repository is built based on the [PySlowFast](https://github.com/facebookresearch/SlowFast).

## License
MViT is released under the [Apache 2.0 license](LICENSE).

## Citation
If you find this repository helpful, please consider citing:
```
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}

@inproceedings{fan2021multiscale,
  title={Multiscale vision transformers},
  author={Fan, Haoqi and Xiong, Bo and Mangalam, Karttikeya and Li, Yanghao and Yan, Zhicheng and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={ICCV},
  year={2021}
}
```
