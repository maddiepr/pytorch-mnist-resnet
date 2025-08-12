"""
evaluate.py

Model evaluation script for the PyTorch MNIST & ResNet Transfer Learning project.

This script handles:
    - Loading a trained model checkpoint from 'outputs/models/'.
    - Initializing the appropriate dataset and DataLoader for evaluation.
    - Computing performance metrics such as accuracy, confusion matrix,
      and classification report.
    - Optionally generating visualization (e.g., loss/accuracy curves, 
      confusion matrix plots, example predictions).
    - Logging results to the console and/or saving them to disk.

Usage:
    python -m src.evaluate --checkpoint outputs/models/mnist_cnn.pt

Notes:
    - This script assumes the experiment configuration used for training
      is available to ensure consistent preprocessing and model setup.
    - Supports evaluation on both CPU and GPU.
"""