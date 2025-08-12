"""
data.py

Data loading and preprocessing utilities for the PyTorch MNIST & ResNet Transfer Learning project.

This module provides functions to:
    - Download and prepare the MNIST data (train/test splits).
    - Optionally load a small custom dataset for transfer learning experiments.
    - Apply appropriate image transformations (e.g., normalization, resizing, augmentation).
    - Wrap datasets in PyTorch DataLoader objects for batched iteration.
    - Ensure reproducibility in dataset shuffling and splitting.

Intended Usage:
    Import from 'train.py', 'evaluate.py', or 'transfer_learning.py' to obtain ready-to-use
    DataLoader instances for training and evaluation.

Example:
    from src.data import get_dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=64, dataset="MNIST)
"""

import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def set_seed(seed: int) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch to ensure reproducible
    experiments.

    Parameters:
        seed: int
            Seed value to apply across all random number generators.

    Notes:
        Reproducibility in deep learning ensures that the same code, data, and
        hyperparameters produce identical results across runs. This is critical
        for debugging, sharing experiments, and scientific validation.
    """
    # Set Python's built-in random module seed
    random.seed(seed)

    # Set NumPy's RNG seed
    np.random.seed(seed)

    # Set PyTorch RNG seed for both CPU and GPU (if available)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure CuDNN (NVIDIA's CUDA Deep Neural Network library) for deterministic behavior
    # - deterministic=True forces deterministic algorithms where possible
    # - benchmark=False prevents CuDNN from selecting the fastest algorithm (which may vary between runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_mnist_transforms(augment: bool = False, rotation_degrees: int = 10) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Preprocessing piplines for MNIST images.

    Parameters:
        augment: bool, optional
            If True, applies light data augmentation (random rotation) to the training
            set to improve generalizaton. Defaults to False.
        rotation_degrees: int, optional
            Maximum degrees for random rotation when augment=True. Defaults to 10.
    
    Returns:
        tuple[transforms.Compose, transforms.Compose]
            A pair of torchvision transform pipelines:
            (train_transforms, test_transforms)
    """
    # Pre-computed mean and standard deviation for MNIST (grayscale)
    # These values are used to normalize pixel intensities so that the
    # model trains more efficiently and converges faster.
    mean, std = (0.1307, ), (0.3081,)

    # Base training transform pipeline: convert to tensor, then normalize.
    train_list = [
        transforms.ToTensor(),              # Convert PIL image or ndarray to PyTorch tensor [C, H, W] in [0, 1]
        transforms.Normalize(mean, std)     # Normalize pixel values channel-wise
    ]

    if augment:
        # Insert light augmentation at the start of the pipeline:
        # Randomly rotate images by +-10 degrees to simulate real handwriting variation
        # and make the model more robust to slight orientation changes.
        train_list.insert(0, transforms.RandomRotation(rotation_degrees))

    # Test transform pipeline: no augmentation, only convert to tensor and normalize.
    # This ensures evaluation is consistent and unbiased.
    test_list = [transforms.ToTensor(),
                transforms.Normalize(mean, std)
    ]

    # Wrap transform lists into callable Compose objects so that they can be passed directly
    # to datasets.
    return transforms.Compose(train_list), transforms.Compose(test_list)

def get_mnist_dataloaders(
        data_dir: str = "data/",
        batch_size: int = 64,
        augment: bool = False,
        num_workers: int = 4,
        seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader objects for the MNIST dataset.

    Parameters:
        data_dir: str, optional
            Directory where MNIST will be downloaded/stored. Defaults to "data/"
        batch_size: int, optional
            Number of samples per batch. Defaults to 64.
        augment: bool, optional
            If True, applies light data augmentation (rotation) to training images.
            Defaults to False.
        num_workers: int, optional
            Number of subprocesses to use for data loading. Defaults to 4.
        seed: int, optional
            Random seed for reproducible shuffling. Defaults to 42.

    Returns:
        tuple[DataLoader, DataLoader]
            (train_loader, test_loader) ready for model training and evaluation.
    """
    # Ensure reproducibility in shuffling and data processing
    set_seed(seed)

    # Get preprocessing pipelines
    train_tfms, test_tfms = get_mnist_transforms(augment=augment)

    # Load MNIST training and test datasets
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_tfms
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=test_tfms
    )

    # Wrap datasets in DataLoader objects for batch processing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,               # Shuffle to break ordering patterns
        num_workers=num_workers,
        pin_memory=True             # Speeds up host-to-GPU data transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,              # No need for shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True            
    )

    return train_loader, test_loader

def get_custom_dataloaders(
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 32,
        augment: bool = True,
        as_grayscale: bool = False,
        num_workers: int = 4,
        seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader objects for a custom image dataset.

    This is intended for transfer learning with pretrained models such as ResNet,
    which expect ImageNet-style normalization.

    Parameters:
        data_dir: str
            Path to the dataset directory. Should contain 'train/' and 'test/' subfolders,
            each with one subfolder per class label.
        image_size: int, optional 
            Target size (height, width) for resizing images. Defaults to 224 for ResNet.
        batch_size: int, optional
            Number of samples per batch. Defaults to 32.
        augment: bool, optional
            If True, applies basic augmentation (random horizontal flip) to training images.
            Defaults to True.
        as_grayscale: bool, optional
            If True, converts images to grayscale before resizing and normalization.
        num_workers: int, optional
            Number of subprocesses to use for data loading. Defaults to 4.
        seed: int, optional
            Random seed for reproducible shuffling. Defaults to 42.

    Returns:
        tuple[DataLoader, DataLoader]
            (train_loader, test_loader) ready for model training and evaluation.
    """
    set_seed(seed)

    # ImageNet mean and std (for pretrained ResNet normalization)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transforms = []
    if as_grayscale:
        train_transforms.append(transforms.Grayscale(num_output_channels=3))    # keeps 3 channels for ResNet
    
    if augment:
        # Insert horizontal flip at start for data augmentation
        train_transforms.append(transforms.RandomHorizontalFlip())

    # Training transforms
    train_transforms.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transforms = []
    if as_grayscale:
        test_transforms.append(transforms.Grayscale(num_output_channels=3))
    
    # Testing transforms (no augmentation)
    test_transforms.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=transforms.Compose(train_transforms)
    )
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=transforms.Compose(test_transforms)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

