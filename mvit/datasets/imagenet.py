# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import json
import os
import random
import re

import mvit.utils.logging as logging
import torch
import torch.utils.data
from mvit.utils.env import pathmgr
from PIL import Image
from torchvision import transforms as transforms_tv

from .build import DATASET_REGISTRY
from .transform import transforms_imagenet_train

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Imagenet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, cfg, mode, num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for ImageNet".format(mode)
        logger.info("Constructing ImageNet {}...".format(mode))
        if cfg.DATA.PATH_TO_PRELOAD_IMDB == "":
            self._construct_imdb()
        else:
            self._load_imdb()

    def _load_imdb(self):
        split_path = os.path.join(
            self.cfg.DATA.PATH_TO_PRELOAD_IMDB,
            f"{self.mode}.json" if self.mode != "test" else "val.json",
        )
        with pathmgr.open(split_path, "r") as f:
            data = f.read()
        self._imdb = json.loads(data)

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self.data_path, self.mode)
        logger.info("{} data path: {}".format(self.mode, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = pathmgr.ls(split_path)
        self._class_ids = sorted(f for f in split_files if re.match(r"^n[0-9]+$", f))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in pathmgr.ls(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im_path):
        with pathmgr.open(im_path, "rb") as f:
            with Image.open(f) as im:
                im = im.convert("RGB")
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )

        if self.mode == "train":
            aug_transform = transforms_imagenet_train(
                img_size=(train_size, train_size),
                color_jitter=self.cfg.AUG.COLOR_JITTER,
                auto_augment=self.cfg.AUG.AA_TYPE,
                interpolation=self.cfg.AUG.INTERPOLATION,
                re_prob=self.cfg.AUG.RE_PROB,
                re_mode=self.cfg.AUG.RE_MODE,
                re_count=self.cfg.AUG.RE_COUNT,
                mean=self.cfg.DATA.MEAN,
                std=self.cfg.DATA.STD,
            )
        else:
            t = []
            if self.cfg.DATA.VAL_CROP_RATIO == 0.0:
                t.append(
                    transforms_tv.Resize((test_size, test_size), interpolation=3),
                )
            else:
                # size = int((256 / 224) * test_size) # = 1/0.875 * test_size
                size = int((1.0 / self.cfg.DATA.VAL_CROP_RATIO) * test_size)
                t.append(
                    transforms_tv.Resize(
                        size, interpolation=3
                    ),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms_tv.CenterCrop(test_size))
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
            aug_transform = transforms_tv.Compose(t)
        im = aug_transform(im)
        return im

    def __load__(self, index):
        try:
            # Load the image
            im_path = self._imdb[index]["im_path"]
            # Prepare the image for training / testing
            if self.mode == "train" and self.cfg.AUG.NUM_SAMPLE > 1:
                im = []
                for _ in range(self.cfg.AUG.NUM_SAMPLE):
                    crop = self._prepare_im(im_path)
                    im.append(crop)
                return im
            else:
                im = self._prepare_im(im_path)
                return im

        except Exception as e:
            print(e)
            return None

    def __getitem__(self, index):
        # if the current image is corrupted, load a different image.
        for _ in range(self.num_retries):
            im = self.__load__(index)
            # Data corrupted, retry with a different image.
            if im is None:
                index = random.randint(0, len(self._imdb) - 1)
            else:
                break
        # Retrieve the label
        label = self._imdb[index]["class"]
        if isinstance(im, list):
            label = [label for _ in range(len(im))]

        return im, label

    def __len__(self):
        return len(self._imdb)
