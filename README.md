# Data Analytics Lab - CA3

Continuous Assessment 3 (Lab CA3) for Data Analytics Laboratory.

## Overview
This repository contains the complete implementation of **Movie Rating Classification using Support Vector Machines (SVM)** on the TMDB Movies dataset (movies.csv).

The goal is to predict whether a movie is **Highly Rated** (Rating $\ge$ 7.0) versus **Below 7.0** based on movie metadata such as runtime, release year, budget, and genre combinations.

## Contents
- **lab_ca3.ipynb**: Complete Jupyter / Google Colab notebook with executed outputs, metrics tables, and visualizations.
- **lab_ca3.py**: Clean, standalone Python implementation of the entire analysis pipeline.

## Pipeline & Methodology
1. **Data Ingestion & Cleaning**:
   - Filter valid movies with at least one vote count and recorded rating.
   - Parse release year from dates.
   - Clean runtime and log-transform positive movie budgets.
   - Create binary target label: highly_rated ($ for vote_average $\ge 7.0$, $ otherwise).

2. **Exploratory Data Analysis**:
   - Rating distribution histogram with a decision cutoff line at .0$.

3. **Preprocessing & Feature Engineering**:
   - Stratified train-test split (\% / 20\%$).
   - Numerical preprocessing pipeline: Median Imputation (SimpleImputer) + Standard Feature Scaling (StandardScaler).
   - Categorical multi-label genre one-hot indicator encoding.

4. **Model Architecture**:
   - **Support Vector Classifier (SVC)** with Radial Basis Function (RBF) kernel.
   - Cost penalty parameter =1.0$, gamma='scale'.
   - Balanced class weighting (class_weight='balanced') to mitigate class imbalance.

5. **Evaluation & Visualization**:
   - Confusion matrix display (True Negatives, False Positives, False Negatives, True Positives).
   - Evaluation metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score.
   - Receiver Operating Characteristic (ROC) curve with Decision Function scores.
   - 2D SVM Decision Boundary visualization over Runtime and Release Year.

## Requirements
To run locally:
`ash
pip install numpy pandas matplotlib scikit-learn
`
