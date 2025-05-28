import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import kagglehub
import io
import joblib

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediksi Telco Customer Churn", layout="wide")
st.title("📱 Prediksi Telco Customer Churn dengan Random Forest")
# Interface Streamlit
tab1, tab2, tab3 = st.tabs(["📝 Load Dataset", "🔍 EDA & Preprocessing", "📊 Model"])

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    return pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def encoding_data(df):
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    encoded_data = encoder.fit_transform(df[categorical_cols + numerical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    final_df = pd.concat([df[['Churn']].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    final_df['Churn'] = final_df['Churn'].map({'No': 0, 'Yes': 1})

    final_df
    return final_df

def sampling_smote(df):
    smote=SMOTE(sampling_strategy='minority') 

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    smote = SMOTE(sampling_strategy='auto', random_state=42)

    X_sampled,y_sampled=smote.fit_resample(X,y)
    return X_sampled, y_sampled

def splitting_scaling(X_sampled,y_sampled):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X_sampled, y_sampled, test_size=0.2, random_state=42, stratify=y_sampled)

    scaler = StandardScaler()
    col_scale = ['num__tenure', 'num__MonthlyCharges', 'num__TotalCharges']

    for col in col_scale:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])
    
    return X_train, X_test, y_train, y_test

if "df" not in st.session_state:
    st.session_state.df = load_data()
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


with tab1:
    st.header("Load Dataset Telco Customer Churn")

    if st.button("Load Dataset", type="primary"):
        st.session_state.df = load_data()
        st.session_state.data_loaded = True
        st.success("Dataset berhasil dimuat!")
        st.subheader("Preview Data")
        st.dataframe(st.session_state.df)

with tab2:
    st.header("EDA & Preprocessing")
    if not st.session_state.get("data_loaded", False):
        st.warning("⚠️ Anda perlu Load Dataset terlebih dahulu pada tab Load Dataset sebelum melakukan EDA & Preprocessing.")
    # st.stop()
    else:
        st.subheader("DataFrame Info")

        if "convert_done" not in st.session_state:
            st.session_state.convert_done = False
        if "encoded_done" not in st.session_state:
            st.session_state.encoded_done = False

        if st.button("🔄 Convert TotalCharges to Float & 🗑️ Drop CustomerID"):
            df = st.session_state.df.copy()
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df['TotalCharges'].fillna(0, inplace=True)
            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"])
            st.session_state.df = df
            st.session_state.convert_done = True
            st.session_state.encoded_done = False
            st.success("Konversi dan penghapusan kolom berhasil!")

        df = st.session_state.df
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.code(info_str, language="python")

        if st.session_state.convert_done:
            st.subheader("Preview DataFrame Sebelum Encoding")
            st.dataframe(df)

            if st.button("🧩 Lakukan Encoding Data"):
                st.subheader("Preview DataFrame Setelah Encoding")
                encoded_df = encoding_data(df)
                st.session_state.encoded_df = encoded_df
                st.session_state.encoded_done = True
                st.success("Encoding berhasil!")


        if st.session_state.convert_done and st.session_state.encoded_done:
            
            encoded_df = st.session_state.encoded_df
            st.write(f"Shape sebelum encoding: {df.shape}")
            st.write(f"Shape setelah encoding: {encoded_df.shape}")

        if "encoded_df" in st.session_state:
            st.subheader("Distribusi Churn")

            if "smote_df" not in st.session_state:
                st.session_state.smote_df = None

            if st.button("🔄 Sampling dengan SMOTE"):
                X_res, y_res = sampling_smote(st.session_state.encoded_df)
                df_resampled = pd.DataFrame(X_res, columns=X_res.columns)
                df_resampled["Churn"] = y_res
                st.session_state.smote_df = df_resampled
                st.success("SMOTE sampling berhasil!")

            if st.session_state.smote_df is not None:
                churn_counts = st.session_state.smote_df["Churn"].value_counts()
                st.info("Menampilkan distribusi setelah SMOTE.")
            else:
                churn_counts = st.session_state.encoded_df["Churn"].value_counts()
                st.info("Menampilkan distribusi data asli.")

            churn_counts.index = churn_counts.index.map({0: "No", 1: "Yes"})
            st.bar_chart(churn_counts)

with tab3:
    st.header("Build Model & Hyperparameter Tuning")
    if st.session_state.get("smote_df") is not None:
        if st.button("Split & Scaling Data"):
            X = st.session_state.smote_df.drop("Churn", axis=1)
            y = st.session_state.smote_df["Churn"]
            X_train, X_test, y_train, y_test = splitting_scaling(X, y)

            # Simpan hasil di session_state agar bisa digunakan nanti
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.success("Data berhasil di-split dan discaling!")
            st.write(f"Jumlah data training: {len(X_train)}")
            st.write(f"Jumlah data testing: {len(X_test)}")
            st.write(f"Proporsi: {len(X_train)/(len(X_train)+len(X_test)):.2%} training / {len(X_test)/(len(X_train)+len(X_test)):.2%} testing")
    else:
        st.warning("⚠️ Anda perlu melakukan SMOTE terlebih dahulu pada tab EDA & Preprocessing sebelum split & scaling data.")