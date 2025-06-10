[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HR2Xz9sU)

# [group2] 股票價格預測

Final Project on 1132. Data Science at NCCU CS

## Goal

This study analyzes major technology stocks in Taiwan by applying various modeling techniques to their historical prices, trading volumes, and technical indicators. The direction of stock price movements are predicted methods such as machine learning and deep learning models, including but not limited to Long Short-Term Memory (LSTM) model and eXtreme Gradient Boosting.

## Contributors

| 組員   | 系級     | 學號      | 工作分配           |
| ------ | -------- | --------- | ------------------ |
| 李柏漢 | 資科碩一 | 113753218 | 簡報組、上台報告   |
| 林靖淵 | 資管碩一 | 113356040 | 程式組、資料前處理 |
| 陳昶安 | 資科碩一 | 113753121 | 簡報組、撰寫文件   |
| 林祐祥 | 資科碩一 | 113753114 | 程式組、資料爬蟲   |
| 廖偉哲 | 資科碩一 | 113753222 | 簡報組、簡報製作   |
| 陳彥融 | 資管四乙 | 110306018 | 程式組、模型訓練   |

## Quick start

#### 1. Python Crawler

Automatically download historical prices and trading data for major Taiwanese technology stocks

```Python
python code/crawler.py
```

#### 2. Data Preprocessing

Integrate, clean, and transform the raw data to generate standardized datasets suitable for deep learning models

```R
Rscript code/preprocessing.R
```

#### 3. LSTM Training and Prediction

Train the LSTM model using the preprocessed data and perform stock price prediction

```R
Rscript code/LSTM.R
```

## Folder organization and its related description

### docs

- This project utilizes the LSTM (Long Short-Term Memory) model for training, which requires the **TensorFlow and keras_tuner packages**. Please ensure that these **essential packages are installed** before execution to guarantee the proper functioning of the model
- [1132_DS-FP_group2.pdf](/docs/Presentation/資料科學期末報告.pdf)
- [1132_DS-Poster_group2.pdf](/docs/Presentation/資料科學海報.pdf)

### data

- Input
  - Source
    - The data used in this study is sourced from the official website of the Taiwan Stock Exchange (TWSE). Historical trading data of major Taiwanese technology stocks are automatically downloaded via the public API. The data includes daily open, high, low, close prices, and trading volumes for individual stocks
  - Format
    - The raw data is stored in CSV format. Data for each stock is segmented by month, with columns including date, open, high, low, close prices, trading volume, trading value, and number of trades
    - After preprocessing, the data is consolidated into a single table with added technical indicators and derived features, and is ultimately saved in CSV format for convenient use in deep learning models
  - Size
    - This study covers 50 major Taiwanese technology stocks, with daily trading data from January 2021 to June 2025. Each stock has approximately 1,000 to 1,100 records, totaling about 50,000 raw data entries. After feature engineering and data splitting, the final training, validation, and test datasets contain approximately 40,000 records
  - docs/台股科技優質股清單.csv
    - major Taiwanese technology stocks list

### code

##### Analysis steps

1. Implement an automated **Python-based crawler** to retrieve historical trading data of target stocks from the Taiwan Stock Exchange (TWSE) official API

- Output Specifications:
  - Dedicated storage folder per stock symbol
  - Monthly segmentation with filename format YYYY_MM.csv
  - Data fields include **date, open, high, low, close prices, and trading volume**

2. Data Preprocessing Pipeline

- Multi-source Integration: Consolidate monthly CSV files into complete **time-series datasets**
- Date format conversion (Republic of China calendar → Gregorian calendar) and outlier detection and correction for numerical fields (Tukey's fences method)
- Calculate 5/20-day Moving Averages (MA), Generate 14-day Relative Strength Index (RSI), Construct MACD technical indicator differentials
- Temporal split into **training (80%), validation (10%), and test (10%) sets**, Z-score standardization parameters derived from training set

3. Modeling

- Prepare the training, validation, and testing datasets in formats appropriate for each type of model
- Perform hyperparameter tuning using either dedicated packages or a custom grid search loop. Train the model on the training set and evaluate its performance on the validation set
- Retrain the final model on the combined training and validation data using the optimal set of hyperparameters

4. Evaluation

- Evaluate the performance of the final model on the testing data
- Compute relevant evaluation metrics, with a primary focus on accuracy in this case
- Compare the results to those of a null model or baseline, and determine whether further tuning is needed or the current model is satisfactory

##### Which method or package do you use?

- Methods of modeling:
1. LSTM Model Construction & Validation
  - Input Layer: 30-day historical data window (30 timesteps × 15 features)
  - Core Layers:
    - 1D Convolutional Layer (filters: 64-192, kernel_size = 3)
    - Bidirectional LSTM Layer (units: 128-384)
  - Output Layer: Binary classification layer with Sigmoid activation
  - Training Protocol:
    - Hyperparameter search via keras-tuner (100 configurations)
    - Weighted cross-entropy loss function for class imbalance mitigation
    - Early stopping mechanism (patience = 6) to prevent overfitting
    - Final model retraining on combined training-validation dataset

2. XGBoost
  - Input format: The date and stock_id columns were removed, and each data point was treated as independent, as XGBoost does not rely on temporal dependencies
  - Training Protocol:
    - A grid search combined with 5-fold cross-validation was performed to identify the optimal set of hyperparameters from a predefined search space
    - The final model was retrained using the selected optimal hyperparameters

3. Logistic Regression
  - Input format: For the same reason as with XGBoost, date and stock_id columns were removed
  - Training Protocol: The model is directly trained on training data

- Packages: keras, tensorflow, keras_tuner, xgboost, dplyr, ggplot2

##### What is a null model for comparison?

- Model :

  - Use a Single variable model to serve as null model, which estimates the probability of a positive class (1 in this case) for each decile bin of a numeric feature, then applies these probabilities to predict new values.
  - Moreover, to identify the variable that best predicts the target, a sweep through all variables in the training dataset was performed.

- Result Comparison :
  - Test accuracy of the null model is approximately **53.70%**
  - The best **LSTM** model and **Logistic regression** achieve accuracies of **55.3% and 56.42%**, respectively
  - According to empirical research, it is normal and reasonable for models such as LSTM and logistic regression to outperform the null model by only 1 to 3 percentage points in stock price movement prediction tasks

### results

- Accuracy of models :
  - LSTM : 0.5532
  - XGBoost : 0.5411
  - Logistic Regression : 0.5642

- Confusion Matrix of models :
  - LSTM
    ![](/results/images/CM_LSTM_0.png)
  - XGBoost
    ![](/results/images/CM_XGB_0.png)
  - Logistic Regression
    ![](/results/images/CM_LogR_0.png)

## References

- [台灣證券交易所](https://www.twse.com.tw/zh/index.html)
- [ChatGPT](https://chatgpt.com/)
- [A Study on Stock Forecasting Using Deep Learning and Statistical Models](https://arxiv.org/html/2402.06689v1)
- [Forecasting stock prices changes using long-short term memory neural network with symbolic genetic programming](https://www.nature.com/articles/s41598-023-50783-0)
- [A Quick Guide to Organizing Computational Biology Projects.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424)
