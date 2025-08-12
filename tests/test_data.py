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
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from src.data import (
    set_seed,
    get_mnist_transforms,
    get_mnist_dataloaders,
    get_custom_dataloaders,
)

# ---------- Unit tests: fast, no downloads --------

def test_get_mnist_transforms_types_and_order():
    rotation_degrees = 15
    train_tfms, test_tfms = get_mnist_transforms(augment=True, rotation_degrees=rotation_degrees)

    # Both should be Compose pipelines
    assert isinstance(train_tfms, transforms.Compose)
    assert isinstance(test_tfms, transforms.Compose)

    # With augment=True, the first transform in train should be RandomRotation
    rot = train_tfms.transforms[0]
    assert isinstance(rot, transforms.RandomRotation)

    # Verify the configured rotation range includes our requested degrees
    # torchvision stores this as (-deg, +deg), cast to floats internally
    assert isinstance(rot.degrees, (tuple, list)) and len(rot.degrees) == 2
    lo, hi = rot.degrees
    assert math.isclose(abs(lo), rotation_degrees, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(abs(hi), rotation_degrees, rel_tol=0, abs_tol=1e-6)

    # Test pipeline should not contain augmentation
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

def _create_dummy_imagefolder_structure(base_dir: Path):
    """
    Create a minimal ImageFolder-compatible dataset structure with
    tiny 8x8 RGB images for testing purposes.

    Structure:
        base_dir/train/classA/img0.png
        base_dir/train/classB/img0.png
        base_dir/test/classA/img0.png
        base_dir/test/classB/img0.png
    """
    for split in ["train", "test"]:
        for class_name in ["classA", "classB"]:
            class_dir = base_dir/split/class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Create one dummy RGB image (8x8 pixels)
            img = Image.new("RGB", (8, 8), color=(255, 0, 0))
            img.save(class_dir/"img0.png")

def test_get_custom_dataloaders_with_dummy_dataset():
    """
    Tests get_custom_dataloaders on a temporary dummy dataset.
    Ensures DataLoader returns batches with correct shapes and no errors.
    """
    from src.data import get_custom_dataloaders

    # Create a temporary directory for fake dataset
    temp_dir = Path(tempfile.mkdtemp())
    try:
        _create_dummy_imagefolder_structure(temp_dir)

        train_loader, test_loader = get_custom_dataloaders(
            data_dir=str(temp_dir),
            image_size=32,    # smaller resize for faster test
            batch_size=2,
            augment=False,
            num_workers=0
        )

        # Pull one batch from train loader
        images, labels = next(iter(train_loader))
        assert images.shape == (2, 3, 32, 32)   # RGB images
        assert labels.shape == (2, )
        assert labels.dtype in (torch.int64, torch.long)

    finally:
        # Cleanup the temporary dataset
        shutil.rmtree(temp_dir)

def test_get_custom_dataloaders_grayscale_flag():
    """
    Ensures the loader works when as_grayscale=True.
    We don't assert channel equality after normalization (means/std differ per channel),
    but we do verify shapes/dtypes and that values are finite.
    """

    def _mk(base: Path):
        for split in ["train", "test"]:
            for cls in ["classA", "classB"]:
                p = base / split / cls
                p.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(0, 255, 0)).save(p / "img.png")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        _mk(temp_dir)
        train_loader, test_loader = get_custom_dataloaders(
            data_dir=str(temp_dir),
            image_size=32,
            batch_size=2,
            augment=False,
            as_grayscale=True,   # NEW flag
            num_workers=0,
            seed=123,
        )
        x, y = next(iter(train_loader))
        assert x.shape == (2, 3, 32, 32)        # still 3 channels for ResNet
        assert y.shape == (2,)
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all()
    finally:
        shutil.rmtree(temp_dir)


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

