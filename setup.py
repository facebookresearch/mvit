# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.


from setuptools import find_packages, setup

setup(
    name="mvit",
    version="0.1",
    author="FAIR",
    url="unknown",
    description="Multiscale Vision Transformers",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "termcolor>=1.1",
        "simplejson",
        "opencv-python",
        "torchvision>=0.4.2",
        "Pillow",
        "sklearn",
        "fairscale",
    ],
    packages=find_packages(exclude=("configs", "tests")),
)
