# ADC-TestDataScience-1

[![image](https://img.shields.io/pypi/v/adc_testdatascience_1.svg)](https://pypi.python.org/pypi/adc_testdatascience_1)

[![image](https://img.shields.io/travis/adiazcarral/adc_testdatascience_1.svg)](https://travis-ci.com/adiazcarral/adc_testdatascience_1)

[![Documentation Status](https://readthedocs.org/projects/adc-testdatascience-1/badge/?version=latest)](https://adc-testdatascience-1.readthedocs.io/en/latest/?version=latest)

TEST 1 -- DATA SCIENCE - CLASIFICACIÓN

-   Free software: MIT license
-   Documentation: <https://adc-testdatascience-1.readthedocs.io>.

⚠️ Important Setup Notes
	1.	Python Version Compatibility
This project requires Python 3.8+. Please ensure you’re using the correct version to avoid compatibility issues.
You can check your Python version with:
```bash
python --version
```

	2.	Add Project Root to PYTHONPATH
To ensure that relative imports work correctly when running scripts, add the project root directory to your PYTHONPATH.
	•	On Linux/macOS:

export PYTHONPATH=$(pwd):$PYTHONPATH


	•	On Windows (CMD):

set PYTHONPATH=%cd%;%PYTHONPATH%


	•	On Windows (PowerShell):

$env:PYTHONPATH = "$(Get-Location);" + $env:PYTHONPATH


This ensures modules inside src/ are properly found when running scripts like:

python src/adc_testdatascience_1/scripts/train_model.py --model=cnn

# 🧠 MNIST Classifier Benchmark

This repository implements and compares several models for classifying (rotated) MNIST digits:

- Logistic Regression
- Convolutional Neural Network (CNN)
- Rotation-Equivariant CNN

The code is modular and includes proper evaluation and clean separation between data, models, and scripts.

---

## ⚙️ Installation & Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```
---

## 🚀 How to Train a Model

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

## ✅ How to Test a Model

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

🧪 Run Unit Tests

To run the unit tests using pytest, make sure you're in the virtual environment and inside the project root folder. Then run:

```bash
pytest tests
```

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

## 🚀 2. Run the container
```bash
docker run -p 8000:8000 adc-model-api
```

The FastAPI server will be available at http://localhost:8000.

🧪 3. Use the API
🔍 Swagger UI

You can test the API easily in your browser:

http://localhost:8000/docs

## 🔮 POST /predict

Send your input data as a JSON list of features:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0.5, 0.3, 0.2, 0.8, 0.1, 0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.3, 0.5, 0.6, 0.2, 0.9, 0.4, 0.1, 0.7, 0.8, 0.3, 0.4, 0.6, 0.2, 0.5, 0.3]]}'
```
Replace the values in "inputs" with the appropriate input vector used during training.

## 📁 Project Structure
The project structure has been captured and is available in the structure.txt file. You can view it using the following command:
```bash
cat structure.txt
```

🧑‍💻 Author

Ángel Díaz Carral

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
