# Bitcoin_Price_Prediction_using_LSTM_with_BayesianOptimization
This project involves predicting Bitcoin prices using a Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN) particularly well-suited for time series forecasting, and furthur optimizing using Bayesian Optimization.

## Project Overview

The aim of this project is to:
1. Perform Exploratory Data Analysis (EDA) on the Bitcoin price dataset.
2. Preprocess the data for training the LSTM model.
3. Train the LSTM model to predict future Bitcoin prices.
4. Evaluate the model using various regression metrics.
5. Visualize the predictions and compare them with the actual prices.

## Dataset

The dataset used in this project is stored in a CSV file named `BTC-USD.csv`. It contains historical data on Bitcoin prices.

## Steps and Methodology

### 1. Data Preprocessing
- **Loading the Dataset:** The dataset is loaded into a pandas DataFrame.
- **Handling Missing Values:** Checking for and handling any missing values in the dataset.
- **Feature Scaling:** Scaling the data using MinMaxScaler to normalize the Bitcoin prices.

### 2. Exploratory Data Analysis (EDA)
- **Yearly Analysis:** Analyzing Bitcoin prices for each year from 2014 to 2024.
- **Overall Analysis:** Performing an overall analysis of the Bitcoin prices from 2014 to 2024.

### 3. Model Building
- **LSTM Model:** Building and training an LSTM model using TensorFlow and Keras. The model consists of multiple LSTM layers, followed by Dense layers for the final prediction.

### 4. Model Evaluation
- **Evaluation Metrics:** Using various regression metrics to evaluate the model's performance, including RMSE, MSE, MAE, explained variance score, R^2 score, mean gamma deviance, and mean Poisson deviance.
- **Prediction and Visualization:** Predicting Bitcoin prices for the next 30 days and visualizing the results.

### 5. Results and Visualization
- **Comparison of Original and Predicted Prices:** Plotting the original Bitcoin closing prices alongside the predicted prices.
- **Future Predictions:** Plotting the last 15 days of the dataset along with the next predicted 30 days.
- **Full Range Visualization:** Visualizing the entire closing stock price with the next 30 days of predicted prices.

## Implementation

### Libraries Used
```python
import os
import pandas as pd
import numpy as np
import math
import datetime as dt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
