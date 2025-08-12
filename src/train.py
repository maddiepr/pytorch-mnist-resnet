"""
train.py

Main training entry point for the PyTorch MNIST & ResNet Transfer Learning project.

This script handles:
    - Parsing experiment configurations (e.g., YAML files in 'experiments/').
    - Initializing datasets and data loaders.
    - Selecting and building the model architecture (Simple CNN or ResNet).
    - Setting up the optimizing, loss function, and learning rate scheduler.
    - Running the training loop for the specified number of epochs.
    - Logging metrics (accuracy, loss) and saving checkpoints to 'outputs/'.

Usage:
    python -m src.train --config experiments/mnist_baseline.yaml

Note:
    This script is designed for reproducibility. Random seeds and experiment configs
    should be fixed to allow exact experiment replication.
"""