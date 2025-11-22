# Bangla Fake News Detection using Machine Learning

A complete machine learning pipeline for detecting **Bangla fake news**, covering dataset preparation, preprocessing, feature engineering, model training, evaluation, and performance analysis.

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Preprocessing Steps](#preprocessing-steps)
7. [Models Used](#models-used)
8. [Evaluation Metrics](#evaluation-metrics)
9. [How to Run](#how-to-run)
10. [Output Files](#output-files)
11. [Future Improvements](#future-improvements)
12. [Citation](#citation)

---

## 🧾 Project Overview

This project develops a **Bangla Fake News Classification** system using multiple supervised machine learning models. The workflow includes data cleaning, Bangla text normalization, TF-IDF-based feature extraction, model comparison, and performance visualization.

The system differentiates between:

* **Authentic Bangla News**
* **Fake Bangla News**

---

## ⭐ Key Features

* Bangla text cleaning and normalization
* TF-IDF vectorization for n-gram ranges
* Multiple ML models with performance comparison
* Confusion matrix and classification report
* Model performance stored in JSON format
* Easily extendable for deep learning

---

## 📂 Dataset Description

The notebook uses two primary datasets:

| File                      | Description                        |
| ------------------------- | ---------------------------------- |
| `LabeledAuthentic-7K.csv` | Contains real Bangla news articles |
| `LabeledFake-1K.csv`      | Contains fake Bangla news articles |

Automatically generated files:

* `ml_performance_{gram}.json` — Model performance summary for each n-gram configuration.

📌 *Make sure these files are placed in the correct folder path used inside the notebook.*

---

## 🗂 Project Structure

```
Bangla_Fake_News_Detection/
│
├── Bangla_Fake_News_Detection_using_Machine_Learning.ipynb
├── LabeledAuthentic-7K.csv
├── LabeledFake-1K.csv
├── ml_performance_{gram}.json       # Auto-generated
└── README.md
```

---

## ⚙️ Installation

Install required dependencies:

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn bidi
```

Optional (if using XGBoost):

```bash
pip install xgboost
```

---

## 🔧 Preprocessing Steps

The notebook performs several Bangla-specific text-cleaning operations:

1. Removing punctuations
2. Removing numbers
3. Lowercasing Bangla text
4. Removing Bangla stopwords
5. Tokenization using NLTK
6. TF-IDF Vectorization
7. Handling class imbalance (if needed)

---

## 🤖 Models Used

Detected models include:

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier (if enabled)
* Multinomial Naive Bayes
* Support Vector Machine (SVM)

You can easily extend this to include LSTM, Bi-LSTM, BERT, BanglaBERT, etc.

---

## 📊 Evaluation Metrics

The notebook uses standard classification metrics:

* **Accuracy Score**
* **Precision**
* **Recall**
* **F1-score**
* **Confusion Matrix**
* **Classification Report**
* **ROC-AUC Score** (if applicable)

Performance for each gram range is stored in:

```
ml_performance_{gram}.json
```

---

## ▶️ How to Run

### 1. Open the Notebook

```
jupyter notebook Bangla_Fake_News_Detection_using_Machine_Learning.ipynb
```

### 2. Ensure datasets are correctly placed

Update the path inside the notebook if needed.

### 3. Run all cells sequentially

Training and evaluation will begin automatically.

---

## 📁 Output Files

| File                            | Purpose                                               |
| ------------------------------- | ----------------------------------------------------- |
| `ml_performance_{gram}.json`    | Stores accuracy/precision/recall/F1 for each ML model |
| Model pickle files (if enabled) | Saved trained models                                  |
| Visualizations (if saved)       | Plots from performance evaluation                     |

---

## 🚀 Future Improvements

* Add deep learning: LSTM, GRU, Bi-LSTM
* Integrate BanglaBERT for improved accuracy
* Use word embeddings: FastText, Word2Vec
* Deploy as a web application (Flask/Streamlit)
* Improve dataset scale and quality

---

## 📚 Citation

If you use this project, please cite:

```
Bangla Fake News Detection using Machine Learning — Jupyter Notebook Implementation (2025)
```

---

If you want, I can also:
✅ Add badges (Python version, License, Stars)
✅ Add screenshots of outputs (confusion matrix, graphs)
✅ Rewrite the README in **Bangla**
Just tell me!
