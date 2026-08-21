"""
Stage 1 pretraining: train DigitCNN on the full MNIST dataset.

Run with:  python3 train.py
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import load_mnist
from model import DigitCNN


def evaluate(model, loader, device):
    """
    Compute classification accuracy on a dataset without updating weights.

    Args:
        model: a DigitCNN instance to evaluate.
        loader: a DataLoader yielding (images, labels) batches.
        device: the torch.device the model and data live on.

    Returns:
        Accuracy as a float between 0 and 1 (correct predictions / total).
    """

    # eval() mode switches off training-only behaviors -- mainly
    # dropout, which should be fully active (not randomly disabled)
    # when we're just measuring performance, not learning.
    model.eval()

    correct = 0
    total = 0

    # no_grad() tells PyTorch not to bother tracking operations for
    # backpropagation here, since we're never going to call
    # .backward() in this function. Pure performance/memory saving --
    # doesn't change the numbers, just skips unnecessary bookkeeping.
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # model(x) gives 10 raw scores per image; argmax picks the
            # index of the highest one as the predicted digit.
            preds = model(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    """
    Run the full Stage 1 pretraining pipeline end to end.

    Loads MNIST, builds a DigitCNN, trains it for a fixed number of
    epochs while tracking validation accuracy, reports final test
    accuracy, and saves the trained weights to
    checkpoints/digit_cnn_mnist.pt.
    """
    # Use a GPU if one's available (much faster for larger models),
    # otherwise fall back to CPU -- which is completely fine for a
    # dataset and model this small.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = load_mnist(
        pickle_path="data/mnist.pkl.gz", batch_size=128
    )

    # Create the model and move its parameters onto whichever device
    # we picked. Model and data both need to live on the same device
    # for PyTorch's operations to work.
    model = DigitCNN().to(device)

    # The optimizer is the algorithm that decides how to adjust each
    # weight based on its gradient. Adam is a reliable, low-maintenance
    # default for problems like this. model.parameters() hands it
    # every learnable weight in the network to manage.
    #
    # lr (learning rate) controls how big each update step is: too
    # high and training becomes unstable/erratic, too low and it
    # crawls. 1e-3 is a very common, safe starting point for Adam
    # specifically.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # The loss function measures how wrong the model's predictions
    # are. CrossEntropyLoss is the standard choice for "pick the
    # right one of N classes" problems -- it compares the model's raw
    # scores against the true label and produces a single number that
    # gets smaller as predictions improve.
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        # train() mode switches dropout back on for the actual
        # learning phase.
        model.train()
        running_loss = 0.0

        # One epoch = one full pass through all 50,000 training
        # images, processed in batches of 128 at a time.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # PyTorch accumulates gradients by default, so we must
            # clear out old ones before computing new ones -- otherwise
            # they'd incorrectly stack up across batches.
            optimizer.zero_grad()

            # Forward pass: run the batch of images through the model
            # to get raw prediction scores.
            logits = model(x)

            # Compare predictions to true labels -- a single number
            # representing how wrong the model was on this batch.
            loss = criterion(logits, y)

            # Backpropagation: PyTorch automatically works out how much
            # each individual weight in the network contributed to this
            # error (using calculus -- the chain rule -- under the
            # hood). This is the "autograd" system; you never derive
            # any of this by hand.
            loss.backward()

            # Actually apply the update: nudge every weight slightly in
            # whichever direction should reduce the loss, sized
            # according to the learning rate.
            optimizer.step()

            # Just for reporting progress below -- not part of the
            # actual learning.
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    test_acc = evaluate(model, test_loader, device)
    print(f"\nfinal test accuracy: {test_acc:.4f}")

    # Save just the learned weights (a "state_dict" -- a dictionary
    # mapping each layer's name to its current tensor of numbers),
    # rather than the whole model object. This is the standard, more
    # portable way to save and later reload a PyTorch model -- exactly
    # what Stage 2 fine-tuning will load back in.
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/digit_cnn_mnist.pt")
    print("saved pretrained weights to checkpoints/digit_cnn_mnist.pt")


if __name__ == "__main__":
    main()
