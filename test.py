import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from tensorflow_addons.layers import InstanceNormalization

import numpy as np
from random import sample
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

from pathlib import Path
import os

from train import *

no_imgs = 2000
input_dim = (128, 128, 3)
# network depth
depth = 4
# kernel size for Conv2D
kernel = 3
# batch_size
n_batch = 16
epochs = 10
# steps per epoch, we have ~1500 samples per domain so calculating steps using it
steps = round(no_imgs / n_batch)

if __name__ == "__main__":
    root = Path("./test/")
    testA, testB = tf_data(root)

    generator_A_B = generator(input_dim, depth, kernel)
    generator_B_A = generator(input_dim, depth, kernel)

    checkpoint = tf.train.Checkpoint(
        generator_A_B=generator_A_B,
        generator_B_A=generator_B_A,
    )
    manager = tf.train.CheckpointManager(checkpoint, "cyclegan", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    x_real_A, _ = generate_real(next(testA), n_batch, 0)
    images_B, _ = generate_fake(x_real_A, generator_A_B, n_batch, 0)

    x_real_B, _ = generate_real(next(testB), n_batch, 0)
    images_A, _ = generate_fake(x_real_B, generator_B_A, n_batch, 0)

    fig, ax = plt.subplots(n_batch, figsize=(75, 75))
    for index, img in enumerate(zip(x_real_B, images_A)):
        concat_numpy = np.clip(np.hstack((img[0], img[1])), 0, 1)
        ax[index].imshow(concat_numpy)
    fig.tight_layout()
