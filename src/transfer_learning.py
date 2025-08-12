"""
transfer_learning.py

Fine-tuning script for transfer learning using a pre-trained ResNet model.

This script handles:
    - Loading a pre-trained ResNet backbone from torchvision.models.
    - Adapting the final classification layer for the target dataset
      (e.g., MNIST or a small custom image dataset).
    - Optionally freezing the backbone layers for feature extraction or
      unfreezing for full fine-tuning.
    - Training the modified model using the same workflow as the baseline
      CNN.
    - Saving training weights and training logs to 'outputs/'.

Features:
    - Supports configurable training parameters (batch size, epochs, learning
      rates, freeze/unfreeze options).
    - Includes reproducibility safeguards (random seeds, saved configs).
    - Works with both CPU and GPU training environments.

Usage:
    python -m src.transfer_learning --data data/custom_small/ \
        --epochs 10 --freeze-backbone True
"""