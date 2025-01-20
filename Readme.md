# CycleGAN" 
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
To use the algorithm a library of images is required as well as a csv-file containing the labels and paths for each image. See `classification.csv` for more information. 

There are two ways of training the classification model.
- Interactive: Load `main.py` as module and create an instance of the `EvoFeatures`-class. Call `load_data` and provide a path to the directory where the csv-file is located. Afterwards call `fit` and provide and output-directory.
- Script: Modify the `if __name__`-section at the end of `main.py` and modify line number 875 to point to the directory with the classification-csv-file. Then run the entire script.

In both cases the training ends with the creation of an `adaboost.pkl` file in the output directory.

## Inference
- Load `main.py` as module and create an instance of the `EnsembleClassifier`-class, providing a path to the previously created adaboost-file.
- Load a set of images as a list of numpy arrays.
- Call the `predict`-method of the `EnsembleClassifier`-class and provide the list of images as argument