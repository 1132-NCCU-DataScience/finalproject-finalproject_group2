[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HR2Xz9sU)

# [group2] 股票價格預測

Final Project on 1132. Data Science at NCCU CS

## Goal

This study analyzes major technology stocks in Taiwan by applying time series modeling to their historical prices, trading volumes, and technical indicators. Stock price prediction is performed using a Long Short-Term Memory (LSTM) model.
The goals of this project.

## Contributors

| 組員   | 系級     | 學號      | 工作分配                           |
| ------ | -------- | --------- | ---------------------------------- |
| 李柏漢 | 資科碩一 | 113753218 | 簡報組、上台報告                   |
| 林靖淵 | 資管碩一 | 113356040 | 程式組、資料前處理                 |
| 陳昶安 | 資科碩一 | 113753121 | 簡報組、撰寫文件、擔任團隊吉祥物🦆 |
| 林祐祥 | 資科碩一 | 113753114 | 程式組、資料爬蟲                   |
| 廖偉哲 | 資科碩一 | 113753222 | 簡報組、簡報製作                   |
| 陳彥融 | 資管四乙 | 110306018 | 程式組、模型訓練                   |

## Quick start

##### 1. Python Crawler

Automatically download historical prices and trading data for major Taiwanese technology stocks.

```Python
python code/crawler.py
```

##### 2. Data Preprocessing

Integrate, clean, and transform the raw data to generate standardized datasets suitable for deep learning models.

```R
Rscript code/preprocessing.R
```

##### 3. LSTM Training and Prediction

Train the LSTM model using the preprocessed data and perform stock price prediction.

```R
Rscript code/LSTM.R
```

## Folder organization and its related description

### docs

- Your presentation, 1132_DS-FP_groupID.ppt/pptx/pdf (i.e.,1132_DS-FP_group2.ppt)
- This project utilizes the LSTM (Long Short-Term Memory) model for training, which requires the **TensorFlow and keras_tuner packages**. Please ensure that these **essential packages are installed** before execution to guarantee the proper functioning of the model.

### data

- Input
  - Source
  - Format
  - Size

### code

##### Analysis steps

1. Implement an automated **Python-based crawler** to retrieve historical trading data of target stocks from the Taiwan Stock Exchange (TWSE) official API.

- Output Specifications:
  - Dedicated storage folder per stock symbol
  - Monthly segmentation with filename format YYYY_MM.csv
  - Data fields include **date, open, high, low, close prices, and trading volume**

2. Data Preprocessing Pipeline

- Multi-source Integration: Consolidate monthly CSV files into complete **time-series datasets**
- Date format conversion (Republic of China calendar → Gregorian calendar) and outlier detection and correction for numerical fields (Tukey's fences method)
- Calculate 5/20-day Moving Averages (MA), Generate 14-day Relative Strength Index (RSI), Construct MACD technical indicator differentials
- Temporal split into **training (80%), validation (10%), and test (10%) sets**, Z-score standardization parameters derived from training set

3. LSTM Model Construction & Validation

- Input Layer: 30-day historical data window (30 timesteps × 15 features)
- Core Layers:
  - 1D Convolutional Layer (filters: 64-192, kernel_size=3)
  - Bidirectional LSTM Layer (units: 128-384)
- Output Layer: Binary classification layer with Sigmoid activation
- Training Protocol:
  - Hyperparameter search via keras-tuner (100 configurations)
  - Weighted cross-entropy loss function for class imbalance mitigation
  - Early stopping mechanism (patience=6) to prevent overfitting
  - Final model retraining on combined training-validation dataset

##### Which method or package do you use?

##### How do you perform training and evaluation?

- Cross-validation, or extra separated data

##### What is a null model for comparison?

### results

- What is your performance?
- Is the improvement significant?

## References

- Packages you use
- Related publications
- [A Quick Guide to Organizing Computational Biology Projects.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000424)
