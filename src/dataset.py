"""
Dataset loading and preprocessing utilities for CIFAR-10.
Supports training, evaluation, and calibration subset extraction for quantization.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=2):
    """
    Returns train and test DataLoaders for CIFAR-10.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_train_transform()
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_test_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def get_calibration_loader(data_dir="./data", num_samples=500, batch_size=64):
    """
    Returns a representative calibration DataLoader sampled from the training set.
    Used for Post-Training Static Quantization (PTQ) activation range estimation.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_test_transform()
    )
    indices = list(range(min(num_samples, len(dataset))))
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
