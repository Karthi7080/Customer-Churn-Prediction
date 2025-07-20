import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE TITLE ---
st.set_page_config(layout="wide")
st.title("Telco Customer Churn Prediction Dashboard")

# --- DATA UPLOAD ---
uploaded_file = st.file_uploader("Upload Telco Churn Dataset (CSV)", type="csv")
if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

# --- DATA CLEANING & ENCODING ---
# Convert 'TotalCharges' to numeric, set errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Drop rows with missing values
df = df.dropna()
# Drop customerID as it's not predictive
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Encode binary columns
binary_mappings = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
for col in ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'Churn']:
    if col in df.columns:
        if col == 'gender':
            df[col] = df[col].map({'Male': 1, 'Female': 0})
        else:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

# Ensure 'SeniorCitizen' is integer
if 'SeniorCitizen' in df.columns:
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

# One-hot encode categorical columns
categorical_cols = [
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
# Only encode columns present in dataframe
categorical_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- DATA PREVIEW SECTION ---
st.header("Data Preview")
st.write(df.head())

# --- EDA SECTION ---
st.header("Exploratory Data Analysis (EDA)")

# Ensure 'Churn' is present and is 0/1
if 'Churn' not in df.columns:
    st.error("'Churn' column not found. Please check your dataset.")
    st.stop()
if df['Churn'].nunique() != 2:
    st.error("'Churn' must have exactly two unique values (0 and 1).")
    st.stop()

# Churn Distribution
st.subheader("Customer Churn Distribution")
fig, ax = plt.subplots(figsize=(6, 3))
sns.countplot(x='Churn', data=df, ax=ax)
plt.title('Churn Distribution (0 = No, 1 = Yes)')
st.pyplot(fig)

churn_rate = df['Churn'].mean()
st.write(f"**Churn Rate:** {churn_rate:.2%}")

# Tenure vs. Churn
if 'tenure' in df.columns:
    st.subheader("Tenure by Churn Status")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='Churn', y='tenure', data=df, ax=ax)
    plt.title('Tenure by Churn Status')
    st.pyplot(fig)
else:
    st.warning("'tenure' column not found; skipping plot.")

# Monthly Charges vs. Churn
if 'MonthlyCharges' in df.columns:
    st.subheader("Monthly Charges by Churn Status")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax)
    plt.title('Monthly Charges by Churn Status')
    st.pyplot(fig)
else:
    st.warning("'MonthlyCharges' column not found; skipping plot.")

# --- MODELING SECTION ---
st.header("Model Training & Evaluation")

if st.button("Train Random Forest Model"):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    st.write("**Accuracy:**", accuracy)
    st.write("**Classification Report:**")
    st.text(cr)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    st.subheader("Top 15 Feature Importances")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    st.pyplot(fig)

# --- CUSTOMER SEGMENTATION (K-MEANS, OPTIONAL) ---
st.header("Customer Segmentation (K-Means Clustering)")

cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
if all(col in df.columns for col in cluster_features):
    if st.button("Run Customer Segmentation"):
        X_cluster = df[cluster_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        st.subheader("Customer Segments by Spending")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Cluster', data=df, palette='viridis', ax=ax)
        plt.title('Customer Segments by Spending (K-Means)')
        st.pyplot(fig)
else:
    st.warning(f"Required columns for K-means clustering ({', '.join(cluster_features)}) not all present; skipping this section.")

# --- END OF SCRIPT ---
