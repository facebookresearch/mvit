# Installation

## Requirements
- Python >= 3.8
- PyTorch >= 1.7, please follow PyTorch official instructions at [pytorch.org](https://pytorch.org)
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
- psutil: `pip install psutil`

## MViT

Clone the MViT repository.
```
git clone https://github.com/facebookresearch/mvit
```

```
cd mvit
python setup.py build develop
```


## Data Preparation

Download the ImageNet-1K classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
