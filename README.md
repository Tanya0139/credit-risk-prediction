# 💳 Credit Risk Predictor

A powerful end-to-end ML pipeline to **predict credit risk** from customer financial data. This solution seamlessly integrates **Apache Spark**, **Delta Lake**, **Azure Data Lake**, and **Streamlit**, delivering interactive visual analytics and real-time predictions.

---

**Dataset:** German Credit Data

<p align="center">
  <img src="https://github.com/Tanya0139/credit-risk-prediction/blob/main/credit.gif" alt="Credit Risk Prediction">
</p>

---

## 🚀 Features

- 🔁 Bronze → Silver → Gold data transformation pipelines
- 🧠 Random Forest Classifier with SHAP-based feature explainability
- 📊 Interactive Streamlit dashboard for metrics and prediction
- ☁️ Native Azure Data Lake Gen2 integration with Delta Lake
- ⚡ Real-time predictions with custom user inputs

---

## 📂 Project Structure

```
.
├── main.py                    # Streamlit App (Train + Predict + Visualize)
├── bronzetosilver.ipynb       # Raw to cleaned data transformation
├── silvertogold.ipynb         # Labeling & final features
├── mount.ipynb                # Azure ADL mount for Spark
```

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/credit-risk-predictor.git
cd credit-risk-predictor

# Install required dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run main.py
```

---

## 📦 Key Code Snippets (with Descriptions)

### 🔌 Connect to Azure Data Lake via Delta Lake

```python
spark = SparkSession.builder \
    .appName("CreditRiskVSCode") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0,...") \
    .config("fs.azure.account.oauth2.client.id", client_id) \
    .config("fs.azure.account.oauth2.client.secret", client_secret) \
    .getOrCreate()

df_spark = spark.read.format("delta").load(gold_path)
df = df_spark.toPandas()
```

---

### 🧹 Preprocess Data (Encoding + Scaling)

```python
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X = df.drop('CreditRisk', axis=1)
y = df['CreditRisk']
X_scaled = scaler.fit_transform(X)
```

---

### 🧠 Train the Model & Evaluate

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

### 📈 Explain Predictions with SHAP

```python
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train, check_additivity=False)
shap.summary_plot(shap_values, features=X.columns)
```

---

### 🌐 Streamlit UI for Real-Time Prediction

```python
input_data = []
for col in X.columns:
    input_val = st.number_input(col) if "_" not in col else st.selectbox(col, [0, 1])
    input_data.append(input_val)

if st.button("Predict Credit Risk"):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    st.success("Prediction: Good Credit Risk" if prediction[0] == 1 else "Bad Credit Risk")
```

---

## 🧠 SHAP Visuals for Explainability

![SHAP Summary](https://shap.readthedocs.io/en/latest/_images/overview.png)

---

## 🚧 Future Enhancements

- Add support for multiple classifiers (e.g., XGBoost, LightGBM)
- Integrate hyperparameter tuning via GridSearchCV or Optuna
- Deploy as Docker container or using Azure App Services
- Add authentication for secure user access
- Add versioned model registry and CI/CD with GitHub Actions

---

## 🛠️ Tech Stack

| Layer        | Tools Used                              |
|--------------|------------------------------------------|
| Language     | Python 3.10+                             |
| ML           | scikit-learn, SHAP                      |
| Visualization| matplotlib, seaborn, Streamlit           |
| Cloud        | Azure Data Lake Gen2 + Delta Lake        |
| Big Data     | Apache Spark                            |

---
---

## UI
Here's a glimpse of the interactive credit risk prediction UI built using Streamlit:
<img width="1000" alt="Screenshot 2025-04-24 at 2 58 56 PM" src="https://github.com/user-attachments/assets/62a09cc3-76b9-4312-bde6-97fd286e8b71" />
<img width="1000" alt="Screenshot 2025-04-24 at 2 59 03 PM" src="https://github.com/user-attachments/assets/7d37202e-59ab-4ecf-9c2c-b056100d4bd4" />
<img width="1000" alt="Screenshot 2025-04-24 at 2 59 14 PM" src="https://github.com/user-attachments/assets/901f8f54-b818-4f97-8bbe-4273448c2e34" />
