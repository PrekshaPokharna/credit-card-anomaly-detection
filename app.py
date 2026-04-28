import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Title
# -------------------------------
st.title("💳 AI Based Credit Card Anomaly Detection System")

st.write("Upload your dataset to detect suspicious transactions using Z-Score and Isolation Forest.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Run Detection
    # -------------------------------
    if st.button("🚀 Run Anomaly Detection"):

        data = df.copy()

        st.write("## 🔧 Data Preprocessing")

        # -------------------------------
        # Handle Missing Values
        # -------------------------------
        data.fillna(data.mean(numeric_only=True), inplace=True)

        # -------------------------------
        # Select Numeric Columns
        # -------------------------------
        numeric_cols = data.select_dtypes(include=np.number).columns
        data_numeric = data[numeric_cols]

        st.write(f"Using {len(numeric_cols)} numeric features for detection.")

        # -------------------------------
        # Scaling
        # -------------------------------
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_numeric)

        # -------------------------------
        # Z-Score Detection
        # -------------------------------
        st.write("## 📐 Z-Score Detection")

        z_scores = np.abs((data_numeric - data_numeric.mean()) / data_numeric.std())
        z_threshold = 3

        data['Z_Anomaly'] = (z_scores > z_threshold).any(axis=1).astype(int)

        # -------------------------------
        # Isolation Forest
        # -------------------------------
        st.write("## 🌲 Isolation Forest")

        iso_model = IsolationForest(contamination=0.02, random_state=42)
        iso_model.fit(scaled_data)

        data['Iso_Pred'] = iso_model.predict(scaled_data)
        data['Iso_Anomaly'] = data['Iso_Pred'].apply(lambda x: 1 if x == -1 else 0)

        # -------------------------------
        # Final Score
        # -------------------------------
        data['Anomaly_Score'] = data['Z_Anomaly'] + data['Iso_Anomaly']

        data['Classification'] = data['Anomaly_Score'].apply(
            lambda x: "Suspicious" if x > 0 else "Normal"
        )

        # -------------------------------
        # Results
        # -------------------------------
        st.subheader("📋 Detection Results")
        st.dataframe(data.head(50))

        # -------------------------------
        # Count Plot
        # -------------------------------
        st.subheader("📈 Normal vs Suspicious")

        fig, ax = plt.subplots()
        sns.countplot(x='Classification', data=data, ax=ax)
        st.pyplot(fig)

        # -------------------------------
        # If your dataset has Amount & Time
        # -------------------------------
        if 'Amount' in data.columns and 'Time' in data.columns:
            st.subheader("📊 Amount vs Time")

            fig2, ax2 = plt.subplots()
            sns.scatterplot(
                x='Time',
                y='Amount',
                hue='Classification',
                data=data,
                ax=ax2
            )
            st.pyplot(fig2)

        # -------------------------------
        # If dataset has actual fraud label
        # -------------------------------
        if 'Class' in data.columns:
            st.subheader("🎯 Model Comparison with Actual Labels")

            comparison = pd.crosstab(data['Classification'], data['Class'])
            st.write(comparison)

        st.success("✅ Detection Completed Successfully!")