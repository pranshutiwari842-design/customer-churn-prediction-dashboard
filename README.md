# Customer Churn Prediction

## 📌 Project Overview

Customer Churn Prediction is a Machine Learning project that identifies customers who are likely to discontinue a service. The project uses telecom customer data to analyze customer behavior, uncover churn patterns, and predict future churn using classification algorithms.

The solution helps businesses improve customer retention by identifying at-risk customers and enabling proactive intervention strategies.

---

## 🎯 Objectives

* Analyze customer behavior and churn trends.
* Perform Exploratory Data Analysis (EDA).
* Build and compare multiple Machine Learning models.
* Predict whether a customer will churn or stay.
* Deploy an interactive web application using Streamlit.

---

## 📂 Dataset

**Dataset:** Telco Customer Churn Dataset

### Features

* Customer Demographics
* Senior Citizen Status
* Partner & Dependents
* Internet Service Type
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges
* Tenure

### Target Variable

* **Churn**

  * Yes = Customer left the service
  * No = Customer retained the service

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Streamlit

---

## 📊 Exploratory Data Analysis

Performed detailed EDA to identify churn patterns:

* Churn by Contract Type
* Churn by Internet Service
* Churn by Payment Method
* Monthly Charges Distribution
* Customer Tenure Analysis
* Correlation Analysis

### Key Findings

* Month-to-month contracts have the highest churn rate.
* Customers with shorter tenure are more likely to churn.
* Electronic check users show higher churn probability.
* Higher monthly charges often correlate with increased churn.

---

## 🤖 Machine Learning Models

### Logistic Regression

A baseline classification model used for interpretability.

### Random Forest Classifier

An ensemble learning model providing improved predictive performance.

### XGBoost Classifier

A gradient boosting algorithm used for advanced classification and feature importance analysis.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

### Sample Results

| Model               | Accuracy | Recall | ROC-AUC |
| ------------------- | -------- | ------ | ------- |
| Logistic Regression | 80%      | 61%    | 84%     |
| Random Forest       | 82%      | 66%    | 87%     |
| XGBoost             | 84%      | 70%    | 89%     |

*Results may vary depending on train-test split and hyperparameter tuning.*

---

## 🚀 Streamlit Web Application

The project includes a Streamlit application where users can:

* Enter customer information
* Predict churn probability
* View prediction results instantly

### Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── churn_model.py
├── app.py
├── churn_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

---

## 💡 Business Impact

Customer churn prediction enables organizations to:

* Improve customer retention strategies
* Reduce revenue loss
* Increase customer lifetime value
* Identify high-risk customers early
* Support data-driven decision-making

---

## 📚 Future Enhancements

* Hyperparameter tuning
* Deep Learning models
* Real-time prediction API
* Cloud deployment
* Customer segmentation integration
  
🔗 Dashboard Link:
https://customer-churn-prediction-dashboard-kxn5f7rycqjpnyp3sd87uk.streamlit.app/
---

## 👨‍💻 Author

**Pranshu Tiwari**

BCA (Data Science) | Aspiring Data Analyst & Data Scientist

### Connect with Me

LinkedIn: https://www.linkedin.com/in/pranshu-tiwari-6113a7350
