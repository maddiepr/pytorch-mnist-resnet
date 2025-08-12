# PyTorch MNIST & ResNet Transfer Learning

End-to-end, reproducible reference implementations for:
- A simple CNN trained from scratch on the MNIST dataset.
- Transfer learning with a pre-trained ResNet model.

## Why this repo?
This project serves as a clean, well-documented example of both training a CNN from scratch and fine-tuning a pre-trained model using PyTorch. It demonstrates best practices in:
- Modular code organization
- Reproducibile experiments
- Model evaluation and visualization

---

## Project Structure

```
pytorch-mnist-resnet/
├─ src/
│ ├─ __init__.py            # Makes src a Python package
│ ├─ data.py                # Data loading & preprocessing
│ ├─ models.py              # CNN architecture & ResNet adaptation
│ ├─ train.py               # Training loop & checkpoint saving
│ ├─ evaluate.py            # Model evaluation & plots
│ └─ transfer_learning.py   # Fine-tuning ResNet
├─ notebooks/               # Jupyter walkthroughs
├─ experiments/             # Config files (YAML)
├─ outputs/                 # Saved models & plots (gitignored)
├─ tests/
│ ├─ test_data.py           # Tests for data.py
├─ requirements.txt
├─ LICENSE 
├─ .gitignore
└─ README.md
```

---

## Installation
```bash
# Clone the repo
git clone https://github.com/maddiepr/pytorch-mnist-resnet.git

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Train a simple CNN on MNIST

```bash
python -m src.train --config experiments/mnist_baseline.yaml
```

Evaluate a saved model
```bash
python -m src.evaluate --checkpoint outputs/models/mnist_cnn.pt
```

Transfer learning with ResNet
```bash
python -m src.transfer_learning --data data/custom_small/ --epochs 10 --freeze-backbone true
```

## Examples Results

TBD

## Repoducibility
- Fixed random seeds
- Saved experiment configs
- Versioned code
- Capture environment in requirements.txt

## Testing
This project includes unit tests for data preprocessing and reproducibility utilities, plus an optional integration test for the MNIST DataLoader.

Run fast, offline tests
```bash
pytest -q
```

Run all tests (including MNIST download)
```bash
RUN_DATA_TEST=1 pytest -q
```

Notes:
- Unit tests validate transform pipelines, normalization math, and reproducible random seeds.
- The integration test downloads MNIST (if not already present in 'data/') and verifies DataLoader shapes and types.

## License
MIT License
