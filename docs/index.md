# Loan Default EDA Documentation

Welcome to the documentation for the **Loan Default EDA and Feature Selection** project.

## Overview
This project provides a production-grade machine learning pipeline to explore mortgage applications, handle target leakage, engineer financial ratios, select discriminative features, and train default prediction models.

## Quick Links
- [Data Dictionary](../references/data_dictionary.md)
- [Master Jupyter Notebook](../notebooks/1.0-eda-and-feature-selection.ipynb)

## Pipeline Commands
- `make data`: Loads raw dataset and verifies integrity.
- `make features`: Runs feature engineering and builds scikit-learn transformation pipelines.
- `make train`: Fits Logistic Regression and Random Forest classifiers.
- `make predict`: Runs batch prediction on processed test data.
