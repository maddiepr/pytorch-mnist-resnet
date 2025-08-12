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

