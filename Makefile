.PHONY: venv data features train predict reports api test lint format clean all

ifeq ($(OS),Windows_NT)
python := python
venv_python := .venv\\Scripts\\python.exe
pip := .venv\\Scripts\\pip.exe
else
python := python3 -u
venv_python := .venv/bin/python3
pip := .venv/bin/pip
endif

## Set up virtual environment and install dependencies
venv:
	$(python) -m venv .venv
	$(pip) install --upgrade pip
	$(pip) install -r requirements.txt pytest ruff

## Verify dataset availability and raw data status
data:
	$(python) -m src.dataset

## Execute feature engineering, preprocessing pipeline, and save train/test artifacts
features:
	$(python) -m src.features

## Train benchmark machine learning models and save model artifact
train:
	$(python) -m src.train

## Run batch prediction on test dataset artifact
predict:
	$(python) -m src.predict

## Generate evaluation metrics (reports/metrics.json) and publication figures (reports/figures/)
reports:
	$(python) -m src.reports

## Run API inference demonstration
api:
	$(python) -m src.api

## Run automated pytest unit & integration test suite
test:
	$(venv_python) -m pytest tests/ -v

## Lint codebase using Ruff
lint:
	$(venv_python) -m ruff check src/ tests/

## Execute full end-to-end pipeline
all: data features train predict reports api

## Clean temporary python build artifacts and bytecode
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
