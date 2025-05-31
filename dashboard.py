import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Load model dan preprocessor ---
preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/best_model.pkl')

# --- Data dummy untuk contoh (bisa ganti dengan df asli kamu) ---
# pastikan dataframe df ada dan sudah didefinisikan, contohnya:
# df = pd.read_csv('data.csv')
# dengan kolom dan tipe sesuai pipeline
import kagglehub

# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)
df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
# Mapping kolom kategori dan numerik (sama seperti di pipeline)
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]
numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Fungsi untuk ambil opsi unik untuk dropdown
def get_unique_sorted(col):
    return sorted(df[col].dropna().unique())

st.title("Aplikasi Prediksi Churn Telco")

st.subheader("Masukkan Data Pelanggan Baru")

input_data = {}

for col in categorical_cols:
    input_data[col] = st.selectbox(f"{col}:", get_unique_sorted(col))

for col in numerical_cols:
    if col == 'tenure':
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        input_data[col] = st.slider(f"{col} (bulan):", min_value=min_val, max_value=max_val, value=min_val, step=1)
    elif col == 'SeniorCitizen':
        input_data[col] = st.radio(f"{col}:", options=[0,1])
    elif col == 'TotalCharges':
        # TotalCharges dihitung dari tenure * MonthlyCharges, jadi skip input manual
        continue
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        input_data[col] = st.slider(f"{col}:", min_value=min_val, max_value=max_val, value=min_val)

# Hitung TotalCharges otomatis
total_charges = input_data['tenure'] * input_data['MonthlyCharges']
st.write(f"**TotalCharges (tenure × MonthlyCharges):** {total_charges:.2f}")
input_data['TotalCharges'] = total_charges

# Tombol prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])

    # Transform data dengan preprocessor
    X_transformed = preprocessor.transform(input_df)

    # Prediksi probabilitas dan kelas
    pred_proba = model.predict_proba(X_transformed)[:, 1][0]
    pred_class = model.predict(X_transformed)[0]

    st.write(f"**Prediksi Churn:** {'Yes' if pred_class==1 else 'No'}")
    st.write(f"**Probabilitas Churn:** {pred_proba:.2f}")