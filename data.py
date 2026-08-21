"""
Loads the MNIST dataset (downloaded as data/mnist.pkl.gz) and wraps it
in PyTorch DataLoaders, which handle batching and shuffling for us.

The pickle stores three splits, each as a (images, labels) pair:
  - images: numpy array, shape (N, 784) -- each image flattened into
    one row of 784 numbers (28 x 28 pixels), values already scaled 0-1
  - labels: numpy array, shape (N,) -- one integer 0-9 per image
"""

import gzip
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# These are the mean and standard deviation of pixel values across the
# entire MNIST training set -- well-known, commonly reused constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def _to_tensor_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    """
    Convert raw MNIST numpy arrays into a PyTorch TensorDataset.

    Reshapes flat image vectors into (channel, height, width) format,
    standardizes pixel values, and converts both images and labels
    into PyTorch tensors of the correct dtype.

    Args:
        images: numpy array of shape (N, 784), pixel values in [0, 1].
        labels: numpy array of shape (N,), integer digit labels 0-9.

    Returns:
        A TensorDataset pairing each standardized image with its label.
    """

    # PyTorch's convolutional layers expect image tensors shaped as
    # (batch, channels, height, width). Our images arrive as flat
    # 784-length vectors, so we reshape each one into (1, 28, 28):
    # 1 channel (grayscale -- a color photo would have 3), 28x28 pixels.
    # The -1 tells numpy "figure out this dimension automatically" --
    # here, it becomes however many images are in this split.
    images = images.reshape(-1, 1, 28, 28).astype(np.float32)

    # Standardize: subtract the mean, divide by the standard deviation.
    # This centers pixel values around 0 with unit variance, which tends
    # to make gradient-based training converge faster and more reliably
    # than leaving raw 0-1 values as-is. Purely a numerical-optimization
    # habit -- doesn't change what information is in the image.
    images = (images - MNIST_MEAN) / MNIST_STD

    # Convert from numpy arrays to PyTorch tensors -- the data structure
    # PyTorch's models and operations actually work with.
    x = torch.from_numpy(images)

    # Labels need to be int64 ("Long" tensors) specifically -- that's
    # the type PyTorch's CrossEntropyLoss expects for class indices.
    y = torch.from_numpy(labels.astype(np.int64))

    # TensorDataset just pairs up each image with its label so they can
    # be indexed and shuffled together as a unit.
    return TensorDataset(x, y)


def load_mnist(pickle_path: str = "data/mnist.pkl.gz", batch_size: int = 128):
    """
    Load the MNIST train/validation/test splits and wrap each in a DataLoader.

    Args:
        pickle_path: path to the gzip-compressed MNIST pickle file.
        batch_size: number of examples per batch for all three loaders.

    Returns:
        A tuple (train_loader, val_loader, test_loader), each a PyTorch
        DataLoader ready to iterate over in a training or evaluation loop.
    """

    # gzip.open because the file is compressed; pickle.load because
    # that's the format Python objects (these numpy arrays) were saved
    # in. encoding="latin1" is needed because this particular pickle
    # was originally written under Python 2.
    with gzip.open(pickle_path, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")

    train_ds = _to_tensor_dataset(*train)
    val_ds = _to_tensor_dataset(*val)
    test_ds = _to_tensor_dataset(*test)

    # DataLoader handles two jobs for us: grouping examples into
    # batches (128 images processed together at a time, rather than
    # one-by-one -- much faster, and gives more stable gradient
    # estimates), and shuffling.
    #
    # We shuffle the training set -- a fresh random order every epoch --
    # so the model can't accidentally learn something from the order
    # examples happen to appear in. We don't shuffle validation/test,
    # since we're only measuring accuracy there, not learning from it;
    # order doesn't matter for that.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
