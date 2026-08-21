"""
Stage 2 fine-tuning: adapt the MNIST-pretrained DigitCNN to real
handwritten digits cropped from bakery invoices (invoice_digits/).

Run with:  python finetune.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data import MNIST_MEAN, MNIST_STD
from model import DigitCNN
from train import evaluate

DIGITS = [str(d) for d in range(10)]


class InvoiceDigitsDataset(Dataset):
    """
    Loads labeled digit crops from invoice_digits/<split>/<digit>/*.png.

    label_tool.py already preprocesses every crop to match MNIST's
    format (grayscale, inverted so ink is light-on-dark, padded to
    square before resizing to 28x28), so the only work left here is
    loading each PNG and applying the same MNIST normalization the
    pretrained model expects.
    """

    def __init__(self, root: str, split: str, augment: bool = False):
        self.augment = augment
        self.samples = []  # list of (path, label)

        for digit in DIGITS:
            digit_dir = os.path.join(root, split, digit)
            if not os.path.isdir(digit_dir):
                continue
            for fname in os.listdir(digit_dir):
                self.samples.append((os.path.join(digit_dir, fname), int(digit)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")

        if self.augment:
            img = _augment(img)

        # Scale to [0, 1] the same way the MNIST pickle was already
        # scaled, then apply MNIST's mean/std so inputs match what the
        # pretrained conv layers were trained on.
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MNIST_MEAN) / MNIST_STD
        x = torch.from_numpy(arr).unsqueeze(0)  # (H, W) -> (1, H, W)

        return x, label


def _augment(img: Image.Image) -> Image.Image:
    """
    Apply a small random rotation and translation to one digit crop.

    Real invoice handwriting varies more in slant and position than
    MNIST's centered digits, and with only a few hundred examples for
    the rarer digits, this gives the model a few synthetic variations
    of each one to train on instead of memorizing the exact crop.
    Kept deliberately small (a few degrees/pixels) so digits stay
    legible and don't cross into looking like a different digit.
    """
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, fillcolor=0)

    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)

    return img


def load_invoice_digits(root: str = "invoice_digits", batch_size: int = 128):
    """
    Load the invoice_digits train/val/test splits and wrap each in a
    DataLoader, mirroring data.py's load_mnist().

    Returns:
        (train_loader, val_loader, test_loader, train_dataset) -- the
        raw train_dataset is also returned so its per-class counts can
        be used to compute loss weights.
    """
    train_ds = InvoiceDigitsDataset(root, "train", augment=True)
    val_ds = InvoiceDigitsDataset(root, "val", augment=False)
    test_ds = InvoiceDigitsDataset(root, "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds


def class_weights(train_ds: InvoiceDigitsDataset, device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    invoice_digits has roughly 4x more '0's than '7'/'8'/'9's (see
    CLAUDE.md), so without weighting the model could get decent
    overall accuracy while barely learning the rare digits. Weights
    are normalized to average ~1 so the overall loss scale stays
    similar to the unweighted case.
    """
    counts = torch.zeros(10)
    for _, label in train_ds.samples:
        counts[label] += 1

    weights = counts.sum() / (counts * len(counts))
    return weights.to(device)


def per_class_accuracy(model, loader, device):
    """
    Break down accuracy by digit instead of just an overall number,
    since class imbalance can hide weak performance on rare digits
    behind a fine-looking aggregate accuracy.
    """
    correct = torch.zeros(10)
    total = torch.zeros(10)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            for d in range(10):
                mask = y == d
                total[d] += mask.sum().item()
                correct[d] += (preds[mask] == d).sum().item()

    return correct, total


def print_per_class_accuracy(correct: torch.Tensor, total: torch.Tensor):
    for d in range(10):
        if total[d] == 0:
            continue
        acc = correct[d] / total[d]
        print(f"  digit {d}: {acc:.4f} ({int(correct[d])}/{int(total[d])})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_ds = load_invoice_digits()

    model = DigitCNN().to(device)
    model.load_state_dict(
        torch.load("checkpoints/digit_cnn_mnist.pt", map_location=device)
    )

    # Zero-shot baseline: how well does the MNIST-only model already do
    # on real invoice digits, before any fine-tuning? This is the
    # number fine-tuning needs to beat.
    zero_shot_acc = evaluate(model, val_loader, device)
    print(f"\nzero-shot val accuracy (before fine-tuning): {zero_shot_acc:.4f}")
    correct, total = per_class_accuracy(model, val_loader, device)
    print_per_class_accuracy(correct, total)

    # Freeze the convolutional feature extractor. It already learned
    # generic stroke/edge/curve detectors from 60k MNIST digits: with
    # only ~2,770 real training crops, letting those layers keep
    # updating risks overfitting to this small dataset and forgetting
    # what Stage 1 learned. Only the fully connected classifier head
    # gets fine-tuned.
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_ds, device))

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/digit_cnn_finetuned.pt"

    epochs = 20
    best_val_acc = 0.0
    print("\nfine-tuning...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        # Keep only the checkpoint with the best validation accuracy
        # seen so far, since with this little data the model can start
        # overfitting well before epoch 20.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\nfinal test accuracy (after fine-tuning): {test_acc:.4f}")
    correct, total = per_class_accuracy(model, test_loader, device)
    print_per_class_accuracy(correct, total)

    print(f"\nsaved best fine-tuned weights to {checkpoint_path}")


if __name__ == "__main__":
    main()
