"""
models.py

Model architectures for the PyTorch MNIST & ResNet Transfer Learning project.

Thhis module provides:
    - A SimpleCNN class for training from scratch on MNIST.
    - A utility function to load a pre-trained ResNet model from torchvision
      and adapt its final layer for a specified number of output classes.
    - Optional helper function to inspect model parameters and architecture
      summaries.

Key design considerations:
    - Keep architectures modular so they can be easily extended or swapped.
    - Maintain compatibility with both CPU and GPU execution.
    - Ensure reproducibility by setting random seeds before weight initialization.

Intended usage:
    Imported by 'train.py' and 'transfer_learning.py' to build the chosen model
    based on experiment configuration.

Example:
    from src.models import SimpleCNN, get_resnet_model
    model = SimpleCNN(num_classes=10)
"""