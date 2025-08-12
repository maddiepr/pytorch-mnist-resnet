"""
src package

Contains the core modules for the PyTorch MNIST & ResNet Transfer Learning project.
Each module implements specific functionality used by the training and evaluation scripts.

Modules:
- data.py: Data loading, preprocessing, and reproducibility utilities.
- models.py: CNN architecture and ResNet transfer learning adaptation.
- train.py: Training loop and checkpoint saving.
- evaluate.py: Model evaluation and visualization.
- transfer_learning.py: Fine-tuning ResNet on a custom dataset.
"""

from .data import set_seed, get_mnist_transforms, get_mnist_dataloaders
# from .models import SimpleCNN, get_resnet_model

__all__ = [
    "set_seed",
    "get_mnist_transforms",
    "get_mnist_dataloaders",
]
