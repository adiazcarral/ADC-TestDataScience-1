.PHONY: help install train test lint format cyclo clean

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  train        Train the model"
	@echo "  test         Run model evaluation"
	@echo "  lint         Run ruff and flake8"
	@echo "  format       Auto-format code (black, isort)"
	@echo "  cyclo        Check cyclomatic complexity"
	@echo "  clean        Remove __pycache__ and .pyc files"

install:
	poetry install

train:
	poetry run python src/adc_testdatascience_1/scripts/train_model.py --model=logistic

test:
	poetry run python src/adc_testdatascience_1/scripts/test_model.py --model=logistic

lint:
	ruff check src tests
	flake8 src tests

format:
	black . && isort .

cyclo:
	radon cc src -a

clean:
	find . -type d -name "__pycache__" -exec rm -r {} \; && find . -name "*.pyc" -delete
