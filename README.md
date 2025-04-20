# ADC-TestDataScience-1

[![image](https://img.shields.io/pypi/v/adc_testdatascience_1.svg)](https://pypi.python.org/pypi/adc_testdatascience_1)

[![image](https://img.shields.io/travis/adiazcarral/adc_testdatascience_1.svg)](https://travis-ci.com/adiazcarral/adc_testdatascience_1)

[![Documentation Status](https://readthedocs.org/projects/adc-testdatascience-1/badge/?version=latest)](https://adc-testdatascience-1.readthedocs.io/en/latest/?version=latest)

TEST 1 -- DATA SCIENCE - CLASIFICACIÓN

-   Free software: MIT license
-   Documentation: <https://adc-testdatascience-1.readthedocs.io>.

## Features

XYZ-TestDataScience-1/
`` bash
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
...
## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
