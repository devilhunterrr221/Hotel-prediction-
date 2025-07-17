
# 🏨 Hotel Booking Demand Prediction

A Machine Learning project to predict hotel booking cancellations using various classification algorithms. The aim is to assist hospitality businesses in understanding customer behavior, optimizing booking policies, and reducing revenue loss due to cancellations.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Project Pipeline](#project-pipeline)
- [Technologies Used](#technologies-used)
- [EDA Highlights](#eda-highlights)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Results & Insights](#results--insights)
- [How to Run the Project](#how-to-run-the-project)
- [Future Scope](#future-scope)
- [Author](#author)

---

## 📖 Project Overview

This project involves analyzing a hotel booking dataset and building multiple machine learning models to predict whether a booking will be canceled or not. A robust prediction system can help hotels optimize resource allocation, manage overbookings, and increase profitability.

---

## ❓ Problem Statement

**Objective**: To predict whether a hotel booking will be **canceled** based on historical data using supervised classification models.

**Target Variable**: `is_canceled` (0: Not Canceled, 1: Canceled)

---

## 📂 Dataset Description

- **Source**: [Kaggle - Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Size**: 32,000+ rows, 32 columns

**Key Columns:**
| Feature Name           | Description |
|------------------------|-------------|
| hotel                  | Resort or City hotel |
| lead_time              | Number of days between booking and arrival |
| arrival_date           | Arrival date details |
| stays_in_weekend_nights| Weekend stay count |
| adults, children, babies | Number of guests |
| meal, country, market_segment | Customer & market info |
| deposit_type           | Deposit paid or not |
| customer_type          | Contract, transient, etc. |
| is_canceled            | **Target** - Booking canceled (1) or not (0) |

---

## 🔁 Project Pipeline

```bash
1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training (Multiple Algorithms)
6. Evaluation & Hyperparameter Tuning
7. Model Comparison
8. Result Interpretation
```

---

## ⚙️ Technologies Used

- Python 3.10+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- joblib (for model serialization)

---

## 📊 EDA Highlights

- High cancellation rates were observed in **City Hotels**.
- **Longer lead times** correlated strongly with cancellations.
- **Repeated guests** had significantly fewer cancellations.
- Significant class imbalance: ~37% cancellations, 63% non-cancellations.

Plots included:
- Correlation heatmap
- Booking trends by hotel type
- Cancellation rate by market segment

---

## 🧱 Feature Engineering

- Date features transformed: combined into `arrival_date`
- Categorical features encoded (OneHot/Label Encoding)
- Created `total_guests` = adults + children + babies
- Dropped highly correlated or redundant columns
- Imputation strategies for missing values (mean, mode)

---

## 🤖 Model Building

The following ML models were trained:

| Model               | Description |
|--------------------|-------------|
| Logistic Regression| Baseline |
| Decision Tree       | Tree-based learning |
| Random Forest       | Ensemble model |
| XGBoost Classifier  | Gradient boosting |
| KNN Classifier      | Distance-based |
| Support Vector Machine | Max-margin classifier |

Each model was trained with:
- Stratified train-test split (80:20)
- Cross-validation (5-fold)
- Scikit-learn pipelines
- Class imbalance handling (SMOTE)

---

## 📈 Model Evaluation

| Metric        | Logistic | RandomForest | XGBoost |
|---------------|----------|--------------|---------|
| Accuracy      | 77%      | 86%          | 88%     |
| Precision     | 72%      | 85%          | 87%     |
| Recall        | 64%      | 88%          | 90%     |
| F1 Score      | 68%      | 86%          | 88%     |
| AUC-ROC       | 0.79     | 0.92         | 0.94    |

*XGBoost showed best performance with high recall, suitable for minimizing false negatives.*

---

## ✅ Results & Insights

- **Lead Time**, **Deposit Type**, and **Customer Type** were key predictors.
- Guests who paid deposits were less likely to cancel.
- Transient bookings and bookings with long lead times had high cancellation rates.
- Final model saved as: `xgboost_hotel_model.joblib`

---

## ▶️ How to Run the Project

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/hotel-booking-prediction.git
   cd hotel-booking-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Use saved model for inference**:
   ```python
   from joblib import load
   model = load('xgboost_hotel_model.joblib')
   prediction = model.predict(X_test)
   ```

---

## 🔮 Future Scope

- Integrate model with real-time hotel booking APIs
- Build a web dashboard using Streamlit
- Apply advanced NLP to analyze special request fields
- Use time-series forecasting for overbooking management

---

## 👤 Author

**Naman Malhotra**  
Technical Analyst | Data Science Enthusiast  
📧 [malhotran.654@gmail.com]  
🌐 • [LinkedIn](https://linkedin.com/in/naman-malhotra-658b212b2/) • [GitHub](https://github.com/devilhunterrr221)

---

> **NOTE**: This project is for educational and interview purposes and uses open-source datasets.
