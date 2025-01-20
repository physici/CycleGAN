# https://medium.com/data-science-in-your-pocket/understanding-cyclegans-using-examples-codes-f5d6e1a47048
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from tensorflow_addons.layers import InstanceNormalization

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple
from random import sample
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

from pathlib import Path
import os

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


def preprocess(records: tf.data.Dataset) -> tf.data.Dataset:
    """
    Pre-processing of the Tensorflow-Datasets by scaling the 8bit-range to
    float numbers between 0 and 1.

    Parameters
    ----------
    records : Tensorflow Dataset
        The training images

    Returns
    -------
    Tensorflow Dataset
        The scaled images.
    """
    images = records["image"]
    images = tf.cast(images, tf.float32) / 255.0
    return images


def tf_pipeline(dataset: List[NDArray[np.uint8]]) -> tf.data.Dataset:
    """
    Create a pre-processing pipeline for training of the algorithm

    Parameters
    ----------
    dataset : List
        List of training images

    Returns
    -------
    Tensorflow Dataset
        Training images in Tensorflow format
    """
    data = tf.data.Dataset.from_tensor_slices({"image": dataset})
    data = data.map(preprocess)
    data = data.repeat().shuffle(100).batch(16).prefetch(1)
    return data


def load_data(root: Path, img_count: int = no_imgs):
    """
    Load training images from hard disk

    Parameters
    ----------
    root : pathlib.Path
        Path to the directory with the training images
    img_count : int, optional
        number of images that should be loaded.

    Returns
    -------
    iterator
        iterators over the loaded images
    """
    trainingA = []
    df = pd.read_csv(root / "meta.csv")
    selection = df.query("type == 'template'")
    files = selection["image_url"].to_list()
    img_count = min(img_count, len(files))
    for x in sample(files, img_count):
        image = io.imread(x)
        image = gray2rgb(image)
        image = resize(image, (128, 128), preserve_range=True)
        trainingA.append(np.array(image))

    trainingB = []
    selection = df.query("type == 'target'")
    files = selection["image_url"].to_list()
    img_count = min(img_count, len(files))
    for x in sample(files, img_count):
        image = io.imread(x)
        image = resize(image, (128, 128), preserve_range=True)
        trainingB.append(np.array(image))

    a, b = tf_pipeline(trainingA), tf_pipeline(trainingB)

    return a.__iter__(), b.__iter__()


def discriminator(
    input_dim: Tuple[int, int, int], depth: int, kernel: int
) -> models.Sequential:
    """
    Discriminator of the GAN

    Parameters
    ----------
    input_dim : Tuple[int, int, int]
        Dimension of the input images: height, width, color channels
    depth : int
        Number of convolution blocks in the discriminator
    kernel : int
        Size of the convolution kernel

    Returns
    -------
    models.Sequential
        Keras sequential model
    """
    my_layers = []
    my_layers.append(layers.Input(shape=input_dim))
    # downsample the image
    # Instance Normalization is required as we do not have pairs of images
    # each images need to be regarded as an individual, not being part of a set
    for i in range(1, depth):
        my_layers.append(layers.Conv2D(16 * i, kernel_size=kernel))
        my_layers.append(InstanceNormalization())
        my_layers.append(layers.Activation("relu"))
        my_layers.append(layers.Dropout(0.2))
    my_layers.append(layers.Conv2D(1, kernel_size=kernel))
    my_model = models.Sequential(my_layers)
    my_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam())
    return my_model


def generator(
    input_dim: Tuple[int, int, int], depth: int, kernel: int
) -> models.Sequential:
    """
    Generator of the GAN

    Parameters
    ----------
    input_dim : Tuple[int, int, int]
        Dimension of the input images: height, width, color channels
    depth : int
        Number of convolution blocks in the discriminator
    kernel : int
        Size of the convolution kernel

    Returns
    -------
    models.Sequential
        Keras sequential model
    """
    my_layers = []
    my_layers.append(layers.Input(shape=input_dim))
    # downsample the image
    # Instance Normalization is required as we do not have pairs of images
    # each images needs to be regarded as an individual, not being part of a set
    for i in range(1, depth):
        my_layers.append(layers.Conv2D(16 * i, kernel_size=kernel))
        my_layers.append(InstanceNormalization())
        my_layers.append(layers.Activation("relu"))
        my_layers.append(layers.Dropout(0.2))

    # upsample the features again
    for i in range(1, depth):
        my_layers.append(layers.Conv2DTranspose(16 * i, kernel_size=kernel))
        my_layers.append(InstanceNormalization())
        my_layers.append(layers.Activation("relu"))
        my_layers.append(layers.Dropout(0.2))

    # ensure that the output data has the shape (128, 128) and a single color channel only
    resizer = lambda name: layers.Lambda(
        lambda images: tf.image.resize(images, [128, 128]), name=name
    )
    my_layers.append(resizer("Reshape"))
    my_layers.append(layers.Conv2DTranspose(3, kernel_size=1, activation=None))
    model = models.Sequential(my_layers)
    return model


def composite_model(
    g1: models.Sequential,
    d: models.Sequential,
    g2: models.Sequential,
    image_dim: Tuple[int, int, int],
):
    """
    That's the Cycle in the CycleGan.
    Forward transformation A->B, using the discriminator and the identity loss.
    Then backward transformation B->A, using a cycle-loss value to estimate performance.

    Parameters
    ----------
    g1 : models.Sequential
        _description_
    d : models.Sequential
        _description_
    g2 : models.Sequential
        _description_
    image_dim : Tuple[int, int, int]
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # we can only train one generator per cycle
    g1.trainable = True
    g2.trainable = False
    d.trainable = False

    # Forward A-> B, Adversarial loss
    input_img = layers.Input(shape=input_dim)
    g1_out = g1(input_img)
    d_out = d(g1_out)

    # Forward A-> B, identity loss
    input_id = layers.Input(shape=input_dim)
    g1_out_id = g1(input_id)

    # Backward B->A
    g2_out = g2(g1_out)

    # Backward B->A, Cycle Loss
    g2_out_id = g2(input_id)
    output_g1 = g1(g2_out_id)

    model = models.Model([input_img, input_id], [d_out, g1_out_id, g2_out, output_g1])
    model.compile(
        loss=["mse", "mae", "mae", "mae"],
        loss_weights=[1, 5, 10, 10],
        optimizer=tf.keras.optimizers.Adam(),
    )
    return model


def generate_real(
    dataset: tf.data.Dataset, batch_size: int, patch_size: int
) -> Tuple[tf.data.Dataset, NDArray[np.float_]]:
    """
    Create labels (real - 1 or fake - 0) for each image to help during training.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Image dataset
    batch_size : int
        Batch size during training
    patch_size : int


    Returns
    -------
    Tuple[tf.data.Dataset, NDArray[np.float]]
        Dataset and labels
    """
    labels = np.ones((batch_size, patch_size, patch_size, 1))
    return dataset, labels


def generate_fake(
    dataset: tf.data.Dataset, g: models.Sequential, batch_size: int, patch_size: int
) -> Tuple[tf.data.Dataset, NDArray[np.float_]]:
    """
    Create labels (real - 1 or fake - 0) for each image to help during training.

    Parameters
    ----------
    dataset : tf.data.Dataset
        Image dataset
    g : models.Sequential
        Generator to create the fake images
    batch_size : int
        Batch size during training
    patch_size : int

    Returns
    -------
    Tuple[tf.data.Dataset, NDArray[np.float]]
        Dataset and labels
    """
    predicted = g(dataset)
    labels = np.zeros((batch_size, patch_size, patch_size, 1))
    return predicted, labels


def train(
    discriminator_A: models.Sequential,
    discriminator_B: models.Sequential,
    generator_A_B: models.Sequential,
    generator_B_A: models.Sequential,
    composite_A_B: models.Sequential,
    composite_B_A: models.Sequential,
    epochs: int,
    batch_size: int,
    steps: int,
    n_patch: int,
) -> None:
    """
    Training of the CycleGAN

    Parameters
    ----------
    discriminator_A : models.Sequential
        Discriminator for the GAN A->B
    discriminator_B : models.Sequential
        Discriminator for the GAN B->A
    generator_A_B : models.Sequential
        Generator for GAN A->B
    generator_B_A : models.Sequential
        Generator for GAN B->A
    composite_A_B : models.Sequential
        CycleGAN A->B
    composite_B_A : models.Sequential
        CycleGAN B->A
    epochs : int
        Number of epochs for training
    batch_size : int
        Batch size per training step
    steps : int
        Number of steps per epoch
    n_patch : int
    """
    # create a checkpoint for safekeeping of the intermediate results
    checkpoint = tf.train.Checkpoint(
        generator_A_B=generator_A_B,
        generator_B_A=generator_B_A,
        discriminator_A=discriminator_A,
        discriminator_B=discriminator_B,
        composite_A_B=composite_A_B,
        composite_B_A=composite_B_A,
    )
    manager = tf.train.CheckpointManager(checkpoint, "checkpoints", max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)

    # that's the actual training
    for epoch in range(1, epochs):
        for step in range(1, steps):
            print(epoch, step)

            x_real_A, y_real_A = generate_real(next(trainA), n_batch, n_patch)
            x_real_B, y_real_B = generate_real(next(trainB), n_batch, n_patch)

            x_fake_A, y_fake_A = generate_fake(
                x_real_B, generator_B_A, n_batch, n_patch
            )
            x_fake_B, y_fake_B = generate_fake(
                x_real_A, generator_A_B, n_batch, n_patch
            )

            g_A_B_loss, _, _, _, _ = composite_A_B.train_on_batch(
                [x_real_A, x_real_B], [y_real_B, x_real_B, x_real_A, x_real_B]
            )
            disc_A_real_loss = discriminator_A.train_on_batch(x_real_A, y_real_A)
            disc_A_fake_loss = discriminator_A.train_on_batch(x_fake_A, y_fake_A)

            g_B_A_loss, _, _, _, _ = composite_B_A.train_on_batch(
                [x_real_B, x_real_A], [y_real_A, x_real_A, x_real_B, x_real_A]
            )
            disc_B_real_loss = discriminator_B.train_on_batch(x_real_B, y_real_B)
            disc_B_fake_loss = discriminator_B.train_on_batch(x_fake_B, y_fake_B)

            print("g_A_B_loss", g_A_B_loss)
            print("g_B_A_loss", g_B_A_loss)

            manager.save()


if __name__ == "__main__":
    print("Load Data")
    root = Path("./train/")
    trainA, trainB = load_data(root)

    discriminator_A = discriminator(input_dim, depth, kernel)
    discriminator_B = discriminator(input_dim, depth, kernel)

    generator_A_B = generator(input_dim, depth, kernel)
    generator_B_A = generator(input_dim, depth, kernel)

    # CycleGAN, for Training generator A->B
    composite_A_B = composite_model(
        generator_A_B, discriminator_B, generator_B_A, input_dim
    )
    # CycleGAN, for Training generator B->A
    composite_B_A = composite_model(
        generator_B_A, discriminator_A, generator_A_B, input_dim
    )

    print("Start Training")
    train(
        discriminator_A,
        discriminator_B,
        generator_A_B,
        generator_B_A,
        composite_A_B,
        composite_B_A,
        epochs,
        n_batch,
        steps,
        discriminator_A.output_shape[1],
    )
