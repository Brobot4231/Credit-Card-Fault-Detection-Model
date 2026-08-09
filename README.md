# 💳 Credit Card Fraud Detection System

> A Machine Learning pipeline that detects fraudulent credit card transactions in real-time using **Random Forest**, **XGBoost**, and **Logistic Regression**, deployed as an interactive **Streamlit** web application.

---

## 📌 Overview

Credit card fraud is a major concern for financial institutions and consumers worldwide. With millions of transactions processed daily, manual detection is infeasible. This project builds an automated **fraud detection system** trained on anonymized credit card transaction data to classify transactions as legitimate or fraudulent with high accuracy and minimal false positives.

**Best Model Accuracy:** `99.95%` (Random Forest)  
**Best ROC-AUC:** `0.9792` (XGBoost)

---

## ✨ Features

- 🔍 **Real-Time Fraud Detection** — Input transaction details and get instant fraud/legitimate prediction
- 📊 **Probability Score** — View confidence level for fraud classification
- ⚖️ **Class Imbalance Handling** — SMOTE oversampling to address highly skewed fraud data
- 📈 **ROC Curve Analysis** — Model comparison with AUC scores for performance evaluation
- 🌲 **Ensemble Power** — Random Forest + XGBoost for robust, high-precision detection
- 💾 **Model Persistence** — Serialized Random Forest model for production inference

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Imbalance Handling** | Imbalanced-Learn (SMOTE) |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib |
| **Environment** | Jupyter Notebook |

---

## 📁 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records:** 284,807 transactions
- **Features:** 30 numerical input features + 1 target variable
- **Time Period:** 2 days of real transaction data

### Feature Description
| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed between each transaction and the first transaction |
| `V1` - `V28` | Principal Components obtained via PCA (anonymized for privacy) |
| `Amount` | Transaction amount |
| `Class` | Target: 1 = Fraud, 0 = Legitimate |

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| **Legitimate (0)** | 284,315 | ~99.83% |
| **Fraud (1)** | 492 | ~0.17% |

> ⚠️ **Highly Imbalanced Dataset:** Fraud cases represent less than 0.2% of all transactions.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
streamlit
joblib
```

---

## 🚀 Usage

### 1. Train the Model
Open `Credit Card Fault Detection.ipynb` in Jupyter Notebook and run all cells to:
- Load & explore the creditcard dataset
- Perform EDA and visualize class imbalance
- Scale `Amount` and `Time` features using StandardScaler
- Split data with stratification (80/20 train-test)
- Apply **SMOTE** to balance the training set
- Train Logistic Regression, Random Forest, and XGBoost
- Evaluate with confusion matrix, classification report, and ROC-AUC
- Compare models via ROC curves
- Save the best model as `random_forest_fraud_model.pkl`

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will start at `http://localhost:8501`

### 3. How to Use
| Step | Action |
|------|--------|
| **1** | Enter transaction features (V1-V28, Amount, Time) |
| **2** | Click **Predict** to classify the transaction |
| **3** | View result: **Legitimate ✅** or **Fraudulent ❌** with probability |

---

## 📊 Model Performance

### Training Pipeline
| Step | Details |
|------|---------|
| **Original Training Set** | 227,845 transactions |
| **After SMOTE** | 454,902 transactions (balanced) |
| **Test Set** | 56,962 transactions |
| **Scaler** | StandardScaler on Amount & Time |

### Model Comparison

| Model | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC |
|-------|----------|-------------------|----------------|------------------|---------|
| **Random Forest** | **99.95%** | **0.8454** | **0.8357** | **0.8410** | **0.9731** |
| **XGBoost** | **99.92%** | **0.7311** | **0.8878** | **0.8018** | **0.9792** |
| Logistic Regression | 97.43% | 0.0581 | 0.9184 | 0.1094 | 0.9698 |

### Confusion Matrix — Random Forest (Best Model)
| | Predicted: Legit | Predicted: Fraud |
|--|------------------|------------------|
| **Actual: Legit** | 56,849 (TN) | 15 (FP) |
| **Actual: Fraud** | 16 (FN) | 82 (TP) |

> **Random Forest** selected as the final model due to the highest accuracy (99.95%) and excellent balance between precision and recall for fraud detection.

---

## 📂 Project Structure

```
credit-card-fraud-detection/
│
├── 📓 Credit Card Fault Detection.ipynb    # Jupyter Notebook (EDA + Model Training)
├── 🖥️ app.py                               # Streamlit Application
├── 🌲 random_forest_fraud_model.pkl        # Serialized Random Forest Model
├── 🖼️ Interface image.png                  # ROC Curves Comparison / App Screenshot
├── 📄 requirements.txt                     # Python Dependencies
└── 📄 README.md                            # Project Documentation
```

---

## 🖼️ Screenshot

![App Interface](Interface%20image.png)

---

## 🔮 Future Scope

- Integrate **real-time transaction API** for live fraud monitoring
- Implement **anomaly detection** with Autoencoders or Isolation Forest
- Add **feature engineering** to extract time-based patterns (hour of day, weekend flags)
- Deploy to **AWS / Heroku / Streamlit Cloud** for production use
- Build a **dashboard** with transaction volume trends and fraud alerts
- Experiment with **deep learning** models (Neural Networks, LSTM for sequential data)
- Add **explainability** using SHAP values to interpret fraud predictions

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/) for the Credit Card Fraud Detection Dataset
- [Scikit-Learn](https://scikit-learn.org/) for the ML toolkit
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting framework
- [Imbalanced-Learn](https://imbalanced-learn.org/) for SMOTE implementation
- [Streamlit](https://streamlit.io/) for the web application framework

---


> **Disclaimer:** This tool is for educational and research purposes only. It is not intended for production financial systems without proper validation and regulatory compliance.
