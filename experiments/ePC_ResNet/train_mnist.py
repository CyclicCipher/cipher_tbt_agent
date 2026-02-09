"""
Validate ePC on MNIST before CIFAR-10.

Confirms that pure ePC (error optimization + gradient descent weight updates)
achieves reasonable accuracy on MNIST. This validates the core algorithm before
tackling the more complex ResNet + CIFAR-10 setting.

Target: ~95% test accuracy (eBPC baseline: 95.74% with Hebbian updates)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from experiments.ePC_ResNet.epc_model import PCE
from experiments.ePC_ResNet.architectures import get_mlp_mnist


def get_mnist_loaders(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


def train_epoch(model, weight_optim, train_loader, device, epoch):
    model.train()
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in pbar:
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)
        batch_size = data.size(0)

        # Phase 1: Inference (optimize errors)
        model(data, target)

        # Phase 2: Weight update (local learning via E_local)
        weight_optim.zero_grad()
        loss = model.compute_weight_loss(data, target, batch_size)
        loss.backward()
        weight_optim.step()

        # Track accuracy
        with torch.no_grad():
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct = (preds == target).sum().item()
            total_correct += correct
            total_samples += batch_size

        pbar.set_postfix(acc=f"{correct/batch_size:.2%}")

    return total_correct / total_samples


def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

    return total_correct / total_samples


def main():
    # Hyperparameters
    iters = 5          # Error optimization steps
    e_lr = 0.01        # SGD learning rate for errors
    w_lr = 0.001       # Adam learning rate for weights
    batch_size = 128
    num_epochs = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("=" * 60)
    print("ePC MNIST Validation")
    print(f"Architecture: [784, 128, 128, 128, 10], ReLU")
    print(f"Inference: SGD errors, T={iters}, e_lr={e_lr}")
    print(f"Learning: Adam weights, w_lr={w_lr}")
    print(f"Output loss: cross-entropy")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print("=" * 60)

    train_loader, test_loader = get_mnist_loaders(batch_size)

    architecture = get_mlp_mnist(hidden_size=128, num_hidden=3)
    model = PCE(architecture, iters=iters, e_lr=e_lr, output_loss='ce').to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    weight_optim = torch.optim.Adam(model.parameters(), lr=w_lr)

    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_acc = train_epoch(model, weight_optim, train_loader, device, epoch)
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train {train_acc:.2%}, Test {test_acc:.2%}")

    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"eBPC baseline: 95.74% (3 epochs, Hebbian updates)")
    print(f"Target: ~95%")


if __name__ == "__main__":
    main()
