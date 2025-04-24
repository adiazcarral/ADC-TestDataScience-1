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
```
---

🚀 How to Train a Model

Run one of the following from the project root:
```bash
python src/adc_testdatascience_1/scripts/train_model.py --model=logistic
python src/adc_testdatascience_1/scripts/train_model.py --model=cnn
python src/adc_testdatascience_1/scripts/train_model.py --model=rotcnn
```
You can also use:
```bash
make train
```
(defaults to training the logistic model)

✅ How to Test a Model

Run one of the following from the project root:
```bash
python src/adc_testdatascience_1/scripts/test_model.py --model=logistic
python src/adc_testdatascience_1/scripts/test_model.py --model=cnn
python src/adc_testdatascience_1/scripts/test_model.py --model=rotcnn
```
You can also use:
```bash
make test_model
```
(defaults to training the rotcnn model)
This will evaluate the selected model on the test dataset and display metrics and a confusion matrix.

🗃️ Module Overview

src/adc_testdatascience_1/utils/data_utils.py

- Loads MNIST dataset
- Creates training, validation (balanced), and test sets
- Allows for fractioned subset training

src/adc_testdatascience_1/models/

- logistic.py: Linear classifier
- cnn.py: Basic convolutional network
- rot_cnn.py: Rotation-equivariant CNN

src/adc_testdatascience_1/scripts/train_model.py

- CLI to train a model
- Saves trained weights under src/adc_testdatascience_1/models/

src/adc_testdatascience_1/scripts/test_model.py

-CLI to load a trained model and evaluate it
- Computes accuracy, precision, recall, F1
- Plots normalized confusion matrix (percentage format)

🧹 Code Quality
```bash
make lint        # flake8
make format      # black + isort
make cyclo     # radon, vulture
```
✅ Requirements

Python 3.8+
See requirements.txt

🐳 Run the Model API with Docker + FastAPI

📦 1. Build the Docker container
```bash
docker build -t adc-model-api .
```

🚀 2. Run the container
```bash
docker run -p 8000:8000 adc-model-api
```

The FastAPI server will be available at http://localhost:8000.

🧪 3. Use the API
🔍 Swagger UI

You can test the API easily in your browser:

http://localhost:8000/docs

🔮 POST /predict

Send your input data as a JSON list of features:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0.5, 0.3, 0.2, 0.8, 0.1, 0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.3, 0.5, 0.6, 0.2, 0.9, 0.4, 0.1, 0.7, 0.8, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3]]}'
```
Replace the values in "inputs" with the appropriate input vector used during training.

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
```

🧑‍💻 Author

Ángel Díaz Carral

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
