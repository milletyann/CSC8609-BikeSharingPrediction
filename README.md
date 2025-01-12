# CSC8609-BikeSharingPrediction

End of course project, Machine Learning Algorithm, Bike Sharing Location Prediction

# Bike-Sharing Prediction Project

This project analyzes and predicts bike rental counts using machine learning techniques from data gathered in Washington D.C. in 2011 and 2012 (https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). 
The dataset contains daily and hourly data, including meteorological and calendary features.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Implemented Models](#implemented-models)
4. [Features and Methods](#features-and-methods)
5. [Evaluation Metrics](#evaluation-metrics)
6. [How to Launch the Experiments](#how-to-launch-the-experiments)
7. [Script Parameters](#script-parameters)

## Overview

The goal of this project is:

- Predict bike rental counts (regression).
- Explore the impact of events (e.g., weather conditions, holidays, ...).
- Compare different classical machine learning models.

## Dataset Description

The datasets used are `day.csv` and `hour.csv`:

- **Target variable**: `cnt` (total count of bike rentals).
- **Features**:
  - **Categorical**: `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`.
  - **Numerical**: `temp`, `atemp`, `hum`, `windspeed`.
  - **Additional for hourly data**: `hr` (hour of the day).

## Implemented Models

The models implemented are:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Bagging
- Boosting (GradientBoostingRegressor and LightGBM)
- Support Vector Regression (SVR) with RBF kernels
- Dummy Regressor (random)

## Methods

- GridSearchCV, RandomizedSearch and nested cross-validation for hyperparameter tuning.
- Learning Curves.
- Evaluation using RMSE, R², and MAPE metrics.

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures prediction error.
- **R² (Coefficient of Determination)**: Measures variance explained by the model.
- **MAPE (Mean Absolute Percentage Error)**: Measures how much the predictions are off from the real values (in percentage).

## How to Launch the Experiments

There is only one script: `bike-prediction.py`. You can launch it from the terminal with various options to control behavior.

### Basic Command:

```bash
python3 bike-prediction.py
```

## Script Parameters

The script `bike-prediction.py` accepts the following parameters (parameters can be combined):

### 1. `--display-plots`

- **Default**: `False`
- **Description**: Tells whether plots (e.g., pairplots, scatterplots) are displayed during the execution of the script.
- **Command**:
```bash
python3 bike-prediction.py --display-plots
```

### 2. `--run-nested-crossval`

- **Default**: `False`
- **Description**: Determines whether to perform nested cross-validation for hyperparameter tuning or not. If unset, precomputed parameters are loaded.
- **Command**:
```bash
python3 bike-prediction.py --run-nested-crossval
```
