# HYBRID SARIMA–DEEP LEARNING MODELS TO FORECAST THE ITALIAN UNEMPLOYMENT RATE

**Date:** 10-01-2026  
**Country:** Italy  
**Data Frequency:** Quarterly  
**Disaggregation:** Gender and age groups  
**Autors:**

This repository contains a methodological framework for the analysis and forecasting of the unemployment rate in Italy using classical time series models and deep learning approaches, including hybrid models.

---

## Authors and Affiliations

**Elka Segura Sánchez**<sup>1</sup>,  
**Carlo Adornetto**<sup>2</sup>,  
**Pier Francesco Perri**<sup>3</sup>  

<sup>1,3</sup> Department of Economics, Statistics and Finance *“Giovanni Anania”*  
<sup>2</sup> Department of Mathematics and Computer Science  

University of Calabria, Arcavacata di Rende (CS), Italy  

---

## Project Description

The aim of this project is to forecast the Italian unemployment rate by combining linear and nonlinear modeling techniques. Classical econometric models are used to capture trend and seasonal components, while recurrent neural networks are employed to model nonlinear dynamics. Hybrid approaches integrate both methodologies to improve forecasting accuracy.

The analysis is performed on quarterly unemployment time series disaggregated by gender and age group.

---

## Modeling Approaches

The repository includes implementations of the following models:

### Classical Time Series Models
- ARIMA
- SARIMA

These models are used to capture linear structures, trend, and seasonality in the unemployment rate.

### Deep Learning Models
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory networks (LSTM)
- Gated Recurrent Units (GRU)

Neural networks are applied to capture nonlinear patterns in the time series.

### Hybrid Models
- SARIMA–LSTM
- SARIMA–GRU

Hybrid models combine SARIMA for modeling linear components and neural networks for learning nonlinear residual structures.

---

## Methodology

### Data Preprocessing
- Loading of quarterly unemployment time series by gender and age group.
- Handling of missing values.
- Normalization of data for neural network training.
- Application of differencing to achieve stationarity when required.

### Exploratory Analysis
- Analysis of seasonal patterns in the data.
- Examination of differences across gender and age groups.
- Visualization of trends and seasonal behavior.

### Classical Modeling
- Identification of ARIMA and SARIMA parameters *(p, d, q)* and *(P, D, Q, s)* using ACF, PACF, AIC, and BIC criteria.
- Model estimation performed separately for each demographic group.

### Neural Network Design
- Architectures: RNN, LSTM, GRU.
- Activation functions:
  - `tanh` in hidden layers
  - `ReLU` in the output layer
- Optimization using the NADAM algorithm.
- Hyperparameter configuration including:
  - Number of epochs
  - Window size
  - Batch size
  - Dropout rate

### Hybrid Modeling Strategy
For both SARIMA–LSTM and SARIMA–GRU:
1. Estimation of a SARIMA model to capture trend and seasonality.
2. Computation of SARIMA residuals.
3. Training of the neural network on residuals using a sliding window approach.
4. Combination of SARIMA forecasts and neural network predictions.
5. Multi-step forecasting for the next four quarters.

---

## Model Evaluation

Forecasting performance is assessed using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Confidence intervals for the evaluation metrics are obtained through bootstrapping techniques.

---

## Outputs

The project generates the following output files:

### SARIMA–GRU
- `unemployment_forecast_metrics_SARIMA_GRU.xlsx`  
- `unemployment_forecast_SARIMA_GRU.xlsx`

### SARIMA–LSTM
- `unemployment_forecast_metrics_SARIMA_LSTM.xlsx`  
- `unemployment_forecast_SARIMA_LSTM.xlsx`

These files contain performance metrics and multi-step forecasts for each gender and age group.

---

## Usage

The scripts are designed to be executed sequentially:
1. Data preprocessing and exploratory analysis.
2. Classical model estimation.
3. Neural network training.
4. Hybrid model forecasting and evaluation.

---
