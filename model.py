"""
A small CNN (convolutional neural network) for classifying 28x28
grayscale digit images into one of 10 classes (0-9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    A small CNN for classifying 28x28 grayscale digit images (0-9).

    Architecture: two convolutional + pooling blocks extract visual
    features, followed by two fully connected layers that turn those
    features into a 10-way class prediction.
    """

    # Every custom PyTorch model subclasses nn.Module. __init__ is
    # where we declare the layers -- the pieces that hold learnable
    # weights. Declaring them here doesn't define how data flows
    # through them yet; that happens in forward() below.
    def __init__(self, num_classes: int = 10):
        """
        Declare all learnable layers used by this model.

        Args:
            num_classes: number of output classes to predict
                (10 digits, by default).
        """
        super().__init__()

        # --- Convolutional layers ---
        # Conv2d(in_channels, out_channels, kernel_size, padding)
        #
        # conv1: takes our 1-channel (grayscale) image and produces 32
        # output "feature maps" -- 32 different learned pattern
        # detectors, each scanning the image with a 3x3 sliding window
        # (kernel_size=3). padding=1 adds a 1-pixel border so the
        # output stays 28x28 instead of shrinking slightly, which is
        # what plain convolution does without padding.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # conv2: takes those 32 feature maps as input and produces 64
        # new ones -- combining conv1's simple strokes/edges into more
        # complex shapes like curves and loops. This layering is what
        # gives CNNs their hierarchical "simple parts -> complex whole"
        # way of understanding images.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # MaxPool2d looks at each 2x2 block of pixels and keeps only
        # the largest value, discarding the rest. This halves both
        # height and width. Two benefits: less computation for later
        # layers, and a bit of tolerance for the exact pixel position
        # of a feature shifting slightly.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dropout (regularization) ---
        # During training, dropout randomly zeroes out this fraction of
        # activations on each forward pass -- a different random subset
        # every time. This forces the network to not over-rely on any
        # single feature, which helps it generalize instead of just
        # memorizing the training images. Dropout is automatically
        # switched off during evaluation (more on that in train.py).
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # --- Fully connected (dense) layers ---
        # After conv1+pool (28->14) and conv2+pool (14->7), each image
        # is now a 7x7 grid with 64 channels -- 64 * 7 * 7 = 3136
        # numbers total per image. fc1 takes all of those and learns to
        # combine them into 128 higher-level features.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # fc2 is the final layer: it maps those 128 features down to
        # exactly num_classes (10) output numbers -- one "score" per
        # possible digit. The highest-scoring one is the prediction.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Run a batch of images through the network and return class scores.

        Args:
            x: input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Raw, un-normalized class scores ("logits") of shape
            (batch_size, num_classes). Softmax is not applied here --
            CrossEntropyLoss handles that internally during training.
        """
        # forward() defines what actually happens when you call
        # model(x) -- the real computation path, using the layers
        # declared above in __init__.

        # F.relu is the activation function applied after each conv
        # layer: it simply zeroes out negative values and passes
        # positive ones through unchanged. This non-linearity is what
        # lets the network learn complex patterns -- stacking layers
        # without it would collapse into one big linear transformation,
        # no matter how many layers you added.
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        x = self.dropout1(x)

        # Linear layers expect flat vectors, not a 3D grid of
        # (channels, height, width) per image. flatten(x, 1) collapses
        # everything from dimension 1 onward into one long vector per
        # image, while leaving dimension 0 (the batch of images) alone.
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        # Note: no softmax here. CrossEntropyLoss (used in train.py)
        # expects these raw, un-normalized scores ("logits") directly
        # and applies softmax internally -- adding it here ourselves
        # would double it up.
        return x
