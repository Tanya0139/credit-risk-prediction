<p align="center">
  <img src="https://github.com/Tanya0139/credit-risk-prediction/blob/main/credit.gif" alt="Credit Risk Prediction" width="640">
</p>
# 💳 Credit Risk Prediction Using Machine Learning

**Dataset:** German Credit Data

---

## 🔍 1. Introduction

Financial institutions face increasing challenges in assessing the creditworthiness of loan applicants. To minimize defaults and maintain financial stability, it's essential to leverage data-driven approaches to predict credit risk accurately. This project uses the **German Credit dataset** to develop a machine learning model that classifies loan applicants as good or bad credit risk.

---

## 🎯 2. Objectives

- Build a predictive model to classify credit risk.
- Identify key factors influencing creditworthiness.
- Develop an interactive UI for credit risk prediction using **Streamlit**.
- Provide actionable insights to improve the credit evaluation process.

---

## 📊 3. Methodology

### 🗃️ Data Overview

- **Dataset:** 1000 samples from the German Credit dataset  
- **Target Variable:** `CreditRisk` (Simulated: `1 = Good`, `0 = Bad`)  
- **Features:** Age, Job, Housing, Credit amount, Duration, Purpose, and account status  

### 🧹 Data Preprocessing

- **Missing Values:** Dropped rows with nulls  
- **Feature Encoding:** One-hot encoding using `pd.get_dummies()`  
- **Feature Scaling:** Used `StandardScaler` to normalize numerical features

```python
X_scaled = scaler.fit_transform(X)
```

- **Target Creation:** Simulated target based on domain-relevant logic

```python
df['CreditRisk'] = np.where((df['Credit amount'] < 5000) & (df['Duration'] < 24), 1, 0)
```

### 🧠 Model Development

- **Algorithm:** `RandomForestClassifier`
  - Handles numeric and categorical data
  - Robust to outliers and overfitting
  - Provides feature importance scores

- **Training:**
  - 80-20 split using stratified sampling

```python
X_train, X_test, y_train, y_test = train_test_split(...)
model.fit(X_train, y_train)
```

---

## 📈 4. Results

### 🧪 Evaluation Metrics

- **Accuracy:** High accuracy  
- **Classification Report:**

```
              precision    recall  f1-score   support
           0       0.95       0.87      0.91       47
           1       0.90       0.97      0.93       53
```

- Displayed in **Streamlit**:

```python
st.dataframe(pd.DataFrame(report).transpose())
```

### 🪄 Feature Importance

- Top predictors: `Credit amount`, `Duration`, `Checking account_status`, etc.  
- Visualized using:

```python
st.bar_chart(feature_importance)
```

---

## 🧠 5. Model Interpretability

### 📊 SHAP Analysis

- Used `shap.TreeExplainer` for interpretation:

```python
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train, check_additivity=False)
shap.summary_plot(shap_values, features=X.columns)
```

- SHAP plots explain how each feature contributes to the prediction.

---

## 🖥️ 6. Streamlit UI for User Prediction

- Interactive UI with sliders and dropdowns for user input  
- Real-time prediction display

```python
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_array)
```

---

## ✅ 7. Why This Approach Was Selected

- **Random Forest:** Handles complex patterns & provides feature importance  
- **SHAP:** Critical for model transparency in financial domains  
- **Streamlit:** Easy-to-use interface for real-time prediction apps

---

## 💡 8. Conclusions & Recommendations

- The model effectively predicts credit risk using relevant attributes  
- SHAP builds trust through transparency  
- **Recommendation:** Use this as a preliminary screening tool to assist analysts

---

## 📦 9. Deliverables

- ✅ Source code and Streamlit app  
- ✅ SHAP visualizations and classification reports  
- ✅ User demo (video) explaining functionality and results  

---

## UI
Here's a glimpse of the interactive credit risk prediction UI built using Streamlit:
<img width="1000" alt="Screenshot 2025-04-24 at 2 58 56 PM" src="https://github.com/user-attachments/assets/62a09cc3-76b9-4312-bde6-97fd286e8b71" />
<img width="1000" alt="Screenshot 2025-04-24 at 2 59 03 PM" src="https://github.com/user-attachments/assets/7d37202e-59ab-4ecf-9c2c-b056100d4bd4" />
<img width="1000" alt="Screenshot 2025-04-24 at 2 59 14 PM" src="https://github.com/user-attachments/assets/901f8f54-b818-4f97-8bbe-4273448c2e34" />
