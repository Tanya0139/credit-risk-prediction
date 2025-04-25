
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# Step 2: Load Data
client_id = "<client-id>"
client_secret = "<client-secret>"
tenant_id = "<tenant-id>"
storage_account_name = "credrisk"
container_name = "gold"

gold_path = f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/"

# Initialize Spark session with necessary packages
spark = SparkSession.builder \
    .appName("CreditRiskVSCode") \
    .config("spark.jars.packages", ",".join([
        "io.delta:delta-core_2.12:2.4.0",
        "org.apache.hadoop:hadoop-azure:3.3.1",
        "com.azure:azure-storage:8.6.6"
    ])) \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config(f"fs.azure.account.auth.type.{storage_account_name}.dfs.core.windows.net", "OAuth") \
    .config(f"fs.azure.account.oauth.provider.type.{storage_account_name}.dfs.core.windows.net",
            "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider") \
    .config(f"fs.azure.account.oauth2.client.id.{storage_account_name}.dfs.core.windows.net", client_id) \
    .config(f"fs.azure.account.oauth2.client.secret.{storage_account_name}.dfs.core.windows.net", client_secret) \
    .config(f"fs.azure.account.oauth2.client.endpoint.{storage_account_name}.dfs.core.windows.net",
            f"https://login.microsoftonline.com/{tenant_id}/oauth2/token") \
    .getOrCreate()

# Read Delta data
df_spark = spark.read.format("delta").load(gold_path)
df = df_spark.toPandas()
# Simulate CreditRisk for testing (remove once real label is available)
df['CreditRisk'] = np.where((df['Credit amount'] < 5000) & (df['Duration'] < 24), 1, 0)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X = df.drop('CreditRisk', axis=1)
y = df['CreditRisk']
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Step 4: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Evaluation
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Step 6: Feature Importance
st.subheader("Feature Importances")
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
st.bar_chart(feature_df.set_index('Feature'))

# Step 7: SHAP Interpretation
st.subheader("SHAP Summary Plot")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train, check_additivity=False)
shap.summary_plot(shap_values, features=X.columns, show=False)
st.pyplot(plt.gcf())

# Step 8: Streamlit UI for User Prediction
st.title("Credit Risk Predictor")

# Create dummy inputs
input_data = []
for col in X.columns:
    if "_" in col:
        input_val = st.selectbox(col, [0, 1])
    else:
        input_val = st.number_input(col, min_value=-5.0, max_value=5.0, value=0.0)
    input_data.append(input_val)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_array)
    st.success(f"Prediction: {'Good Credit Risk' if prediction[0] == 1 else 'Bad Credit Risk'}")
