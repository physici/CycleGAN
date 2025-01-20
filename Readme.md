# CycleGAN
This repo provides an example for a Generative Adversarial Network that is trained to create synthetic images for an segmentation algorithm. The used CycleGAN-algorithm learns an image-to-image transformation the tries to minimize the loss in the forward and backward conversion of the image.

The provided code follows the following medium-arcticle:
https://medium.com/data-science-in-your-pocket/understanding-cyclegans-using-examples-codes-f5d6e1a47048


## Installation
The repo is managed with poetry. To install the environment run
```
    poetry install
```
from the root directory.

## Usage
The algorithm is split into a training and a test part. Training requires template and target images to be located in a directory `train`. Additionally, a `meta.csv` file is required that contains the respective image urls to differentiate between template and target images. An example file is provided in the `train`-directory

Run `train.py` to start training the CylceGAN.
