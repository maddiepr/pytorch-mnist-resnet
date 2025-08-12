"""
test_data.py

Unit and integration tests for data.py utilities in the PyTorch MNIST
& ResNet Transfer Learning project.

Covers:
    - get_mnist_transforms: Checks transform types, ordering, and normalization math.
    - set_seed: Ensures reproducible RNG behavior.
    - get_mnist_dataloaders: Verifies DataLoader shapes and types.

To run:
    # Fast, offline tests only:
    pytest -q

    # Include MNIST download & DataLoader integration test:
    RUN_DATA_TESTS=1 pytest -q
"""

import os
import math
from PIL import Image 
import torch
from torchvision import transforms 

from src.data import (
    set_seed,
    get_mnist_transforms,
    get_mnist_dataloaders,
)

# ---------- Unit tests: fast, no downloads --------

def test_get_mnist_transforms_types_and_order():
    train_tfms, test_tfms = get_mnist_transforms(augment=True)

    # Both should be Compose pipelines
    assert isinstance(train_tfms, transforms.Compose)
    assert isinstance(test_tfms, transforms.Compose)

    # With augment=True, the first transform in train should be RandomRotation
    assert isinstance(train_tfms.transforms[0], transforms.RandomRotation)

    # Test pipeline should not start with augmentation
    assert not any(isinstance(t, transforms.RandomRotation) for t in test_tfms.transforms)

def test_normalization_on_dummy_image():
    """
    Apply the MNIST train transforms (no augmentation) to a dummy black image.
    After ToTensor(), pixels are 0. After Normalize(mean=0.1307, std=0.3081)
    each pixel becomes (0 - mean)/std = -mean/std
    """
    mean, std = 0.1307, 0.3081
    target_value = -mean / std

    # Use train transformations without augmentation to know exact ops
    train_tfms, _ = get_mnist_transforms(augment=False)

    # Create a 28x28 grayscale ("L") blank image
    img = Image.new("L", (28, 28), color=0)

    x = train_tfms(img) # shape [1, 28, 28], dtype float32
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 28, 28)
    assert x.dtype == torch.float32

    # All pixels should be very close to the expected normalized constant
    assert torch.allclose(x, torch.full_like(x, target_value), atol=1e-6)

def test_set_seed_reproducibility_basic():
    """
    Verifies that set_seed makes PyTorch RNG reproducible for basic rand().
    """
    set_seed(123)
    a = torch.rand(3, 3)

    set_seed(123)
    b = torch.rand(3, 3)

    assert torch.allclose(a, b)


# ----------- Integration test ----------
# Enable by running: RUN_DATA_TESTS=1 pytest-q
import pytest

@pytest.mark.skipif(os.environ.get("RUN_DATA_TESTS", 0) != "1",
                    reason="Set Run_DATA_TEST=1 to run MNIST download test.")
def test_get_mnist_dataloaders_batch_test():
    """
    Smoke test: builds MNIST dataloaders and pulls one batch.
    Skipped by default to keep CI fast and avoid network flakiness.    
    """
    batch_size = 16
    train_loader, test_loader = get_mnist_dataloaders(
        data_dir="data/",
        batch_size=batch_size,
        augment=True,
        num_workers=0,   # keep tests stable across environments
        seed=42,
    )

    images, labels = next(iter(train_loader))
    assert images.shape == (batch_size, 1, 28, 28)
    assert labels.shape == (batch_size, )
    assert labels.dtype in (torch.int64, torch.long)

    # Values are normalized; not strickly bounded to [0, 1] anymore.
    # Just assert no NaNs/Infs.
    assert torch.isfinite(images).all()