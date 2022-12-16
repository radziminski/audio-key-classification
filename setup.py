#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.0",
    description="Audio Key Classification",
    author="Jan Radziminski, Kacper Kamieniarz",
    author_email="",
    url="https://github.com/radziminski/audio-key-classification",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
)
