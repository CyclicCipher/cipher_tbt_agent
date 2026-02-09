"""
Train ePC ResNet-18 on CIFAR-10.

Uses PCESkipConnection with ResNet-18 architecture matching the ePC paper
(Goemaere et al. 2025). Two-phase training: SGD error optimization for
inference, then local weight updates via E_local with Adam optimizer.

Target: 92.17% test accuracy (ePC paper)
Backprop baseline: 92.36%

Reference hyperparameters (from ePC repo cifar branch):
  batch_size=256, epochs=50, iters=5, e_lr=0.001
  w_lr=0.0001, w_decay=0.0, output_loss='mse'
  LR schedule: linear warmup (10%) + cosine decay
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from experiments.ePC_ResNet.epc_model import PCESkipConnection
from experiments.ePC_ResNet.architectures import get_resnet18_cifar10


def get_cifar10_loaders(batch_size=256, data_dir='./data', num_workers=2):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   drop_last=True, num_workers=num_workers),
        DataLoader(test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


def make_lr_schedule(optimizer, total_steps, base_lr=0.0001, warmup_fraction=0.1):
    """Cosine decay with linear warmup (matching ePC paper)."""
    peak_lr = 1.1 * base_lr
    end_lr = 0.1 * base_lr
    warmup_steps = int(warmup_fraction * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return base_lr + (peak_lr - base_lr) * (step / max(1, warmup_steps))
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return end_lr + (peak_lr - end_lr) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_epoch(model, weight_optim, lr_scheduler, train_loader, device, epoch):
    model.train()
    total_correct = 0
    total_samples = 0
    total_energy = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)

        # Phase 1: Inference (optimize errors)
        energy = model(data, target)

        # Phase 2: Weight update (local learning via E_local)
        weight_optim.zero_grad()
        loss = model.compute_weight_loss(data, target, batch_size)
        loss.backward()
        weight_optim.step()
        lr_scheduler.step()

        total_energy += energy

        # Track accuracy
        with torch.no_grad():
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct = (preds == target).sum().item()
            total_correct += correct
            total_samples += batch_size

        pbar.set_postfix(
            acc=f"{correct/batch_size:.1%}",
            lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
        )

    return total_correct / total_samples, total_energy / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

    return total_correct / total_samples


def main():
    # Hyperparameters (matching ePC paper)
    iters = 5           # Error optimization steps
    e_lr = 0.001        # SGD learning rate for errors
    w_lr = 0.0001       # Base learning rate for weights (Adam)
    w_decay = 0.0       # Weight decay
    batch_size = 256
    num_epochs = 50
    output_loss = 'mse'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("=" * 60)
    print("ePC ResNet-18 on CIFAR-10")
    print(f"Inference: SGD errors, T={iters}, e_lr={e_lr}")
    print(f"Learning: Adam, w_lr={w_lr}, w_decay={w_decay}")
    print(f"Output loss: {output_loss}")
    print(f"LR schedule: warmup 10% + cosine decay")
    print(f"Batch size: {batch_size}, Epochs: {num_epochs}")
    print(f"Target: 92.17% (ePC paper)")
    print("=" * 60)

    train_loader, test_loader = get_cifar10_loaders(batch_size)

    architecture = get_resnet18_cifar10()
    model = PCESkipConnection(
        architecture, iters=iters, e_lr=e_lr, output_loss=output_loss,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Adam with lr=1.0 (actual LR controlled by scheduler)
    weight_optim = torch.optim.Adam(
        model.parameters(), lr=1.0, weight_decay=w_decay,
    )
    total_steps = len(train_loader) * num_epochs
    lr_scheduler = make_lr_schedule(weight_optim, total_steps, base_lr=w_lr)

    best_test_acc = 0.0
    for epoch in range(num_epochs):
        train_acc, avg_energy = train_epoch(
            model, weight_optim, lr_scheduler, train_loader, device, epoch,
        )
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch {epoch+1:3d}/{num_epochs}: "
              f"Train {train_acc:.2%}, Test {test_acc:.2%}, "
              f"Energy {avg_energy:.1f}, Best {best_test_acc:.2%}")

    print(f"\nBest test accuracy: {best_test_acc:.2%}")
    print(f"ePC paper target: 92.17%")
    print(f"Backprop baseline: 92.36%")


if __name__ == "__main__":
    main()
