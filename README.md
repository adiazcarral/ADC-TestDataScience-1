# ADC-TestDataScience-1

[![image](https://img.shields.io/pypi/v/adc_testdatascience_1.svg)](https://pypi.python.org/pypi/adc_testdatascience_1)

[![image](https://img.shields.io/travis/adiazcarral/adc_testdatascience_1.svg)](https://travis-ci.com/adiazcarral/adc_testdatascience_1)

[![Documentation Status](https://readthedocs.org/projects/adc-testdatascience-1/badge/?version=latest)](https://adc-testdatascience-1.readthedocs.io/en/latest/?version=latest)

TEST 1 -- DATA SCIENCE - CLASIFICACIÓN

-   Free software: MIT license
-   Documentation: <https://adc-testdatascience-1.readthedocs.io>.

# 🧠 MNIST Classifier Benchmark

This repository implements and compares several models for classifying (rotated) MNIST digits:

- Logistic Regression
- Convolutional Neural Network (CNN)
- Rotation-Equivariant CNN

The code is modular and includes proper evaluation, logging, and clean separation between data, models, and scripts.


---

## ⚙️ Installation & Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

---

🚀 How to Train a Model

Run one of the following from the project root:

python src/adc_testdatascience_1/scripts/train_model.py --model=logistic
python src/adc_testdatascience_1/scripts/train_model.py --model=cnn
python src/adc_testdatascience_1/scripts/train_model.py --model=rotcnn

You can also use:

make train
(defaults to training the logistic model)

🗃️ Module Overview

src/adc_testdatascience_1/data/dataloaders.py
Loads MNIST dataset
Creates training, validation (balanced), and test sets
Allows for fractioned subset training
src/adc_testdatascience_1/models/
logistic.py: Linear classifier
cnn.py: Basic convolutional network
rot_cnn.py: Rotation-equivariant CNN
src/adc_testdatascience_1/evaluation/evaluator.py
Computes accuracy, precision, recall, F1
Plots normalized confusion matrix (percentage format)
Compares multiple models side by side
src/adc_testdatascience_1/scripts/train_model.py
CLI to train a model
Saves trained weights under src/adc_testdatascience_1/models/
src/adc_testdatascience_1/scripts/test_model.py
CLI to load a trained model and evaluate it

🧹 Code Quality

make lint        # flake8
make format      # black + isort
make quality     # radon, vulture

✅ Requirements

Python 3.8+
See requirements.txt
🧑‍💻 Author

Ángel Díaz Carral

## 📁 Project Structure

```bash
│
├── README.md                        # Overview of the project and repo structure
├── requirements.txt                 # All Python dependencies
├── pyproject.toml                   # (Optional) For Poetry-based dependency management
├── .flake8                          # Linter configuration
├── .gitignore                       # Ignore common temp files, logs, cache, etc.
│
├── data/
│   └── raw/                         # Original QM7b data
│   └── processed/                   # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 01_eda_data_analysis.ipynb   # Exploratory data analysis + preprocessing
│   ├── 02_training_validation.ipynb # Model training and validation
│   └── 03_evaluation_testing.ipynb  # Final model evaluation and insights
│
├── src/
│   ├── __init__.py
│   ├── config/                      # Config classes or settings
│   │   └── paths.py
│   ├── data/                        # Loading, preprocessing, graph builders
│   │   ├── load_qm7b.py
│   │   └── preprocess.py
│   ├── models/                      # Model definitions, training loop, utils
│   │   ├── base_model.py
│   │   ├── mlp.py                   # Shallow or FFN model
│   │   └── gnn.py                   # GNN (e.g., GCN, GIN) with PyTorch Geometric
│   ├── training/                    # Training and evaluation pipeline
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/                       # Common helpers (metrics, logging, plotting)
│       ├── metrics.py
│       └── visualizations.py
│
├── models/
│   └── saved_model.pt               # Best trained model (optional, for inference)
│
├── reports/
│   ├── figures/                     # Plots, EDA figures, confusion matrices
│   └── results_summary.pdf          # Summary of results and insights
│
└── docs/
    └── index.md                     # (Optional) If using MkDocs for documentation


## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
