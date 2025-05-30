import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
st.title("📱 Prediksi Telco Customer Churn dengan XGBoost")

tab1, tab2, tab3 = st.tabs(["📝 Load Dataset", "🔍 EDA & Preprocessing", "📊 Model"])

random_state = 42
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    return pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def split_data(X,y):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scaling_data(df, X_train, X_test):
    from sklearn.preprocessing import MinMaxScaler

    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test

def encoding_data(X_train,X_test):
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

    encoded_data = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    X_test = pd.DataFrame(X_test, columns=encoder.get_feature_names_out())
    return encoded_df,X_test

def sampling_data(df):
    from imblearn.under_sampling import RandomUnderSampler

    sampling_method = RandomUnderSampler(sampling_strategy=1.0,random_state=random_state)

    X_sampled,y_sampled=sampling_method.fit_resample(encoded_df,y_train)
    sampled_df = pd.concat([
        pd.DataFrame(X_sampled, columns=encoded_df.columns),
        pd.Series(y_sampled.values, name='Churn')
    ], axis=1)
    return sampled_df

def evaluate_thresholds(X_sampled, y_sampled, X_test, y_test, importance_df, thresholds=None):
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

    results = []

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    best_params = joblib.load('models/grid_search.pkl').best_params_
    for t in thresholds:
        selected = importance_df[importance_df['importance'] >= t]['feature']
        if len(selected) < 2:
            continue

        X_train_sel = X_sampled[selected]
        X_test_sel = X_test[selected]

        model_thresh = XGBClassifier(**best_params)

        model_thresh.fit(X_train_sel, y_sampled)
        y_pred_sel = model_thresh.predict(X_test_sel)
        test_acc = accuracy_score(y_test, y_pred_sel)

        cv_scores = cross_val_score(model_thresh, X_train_sel, y_sampled, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            'Threshold': t,
            'Num_Features': len(selected),
            'Test_Accuracy': test_acc,
            'CV_Accuracy_Mean': cv_mean,
            'CV_Accuracy_Std': cv_std
        })

    results_df = pd.DataFrame(results)
    return results_df

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
        st.subheader("🔹 Preview Data")
        st.dataframe(st.session_state.df)

with tab2:
    st.header("EDA & Preprocessing")
    if not st.session_state.get("data_loaded", False):
        st.warning("⚠️ Anda perlu Load Dataset terlebih dahulu pada tab Load Dataset sebelum melakukan EDA & Preprocessing.")
    else:
        st.subheader("🔹 DataFrame Info")

        if "convert_done" not in st.session_state:
            st.session_state.convert_done = False
        if "split" not in st.session_state:
            st.session_state.split = False
        if "scaled" not in st.session_state:
            st.session_state.scaled = False

        if st.button("🔄 Konversi TotalCharges & 🗑️ Hapus CustomerID"):
            df = st.session_state.df.copy()

            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"])

            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df.dropna(subset=['TotalCharges'], inplace=True)

            st.session_state.df = df
            st.session_state.convert_done = True
            st.success("Konversi dan penghapusan kolom berhasil!")

        df = st.session_state.df
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.code(info_str, language="python")

        if st.session_state.convert_done:
            st.subheader("🔹 Split Dataset")
            if st.button("✂️ Split Dataset"):
                X = st.session_state.df.drop("Churn", axis=1)
                y = st.session_state.df['Churn']
                X_train,X_test,y_train,y_test = split_data(X,y)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.split = True
                st.success("Dataset berhasil di-split!")
                st.write(f"Jumlah data latih: {len(X_train)}")
                st.write(f"Jumlah data uji: {len(X_test)}")
                st.write(f"Proporsi: 80% latih / 20% uji")
        
        if st.session_state.split:
            st.subheader("🔹 Scaling Dataset")

            if st.button("🚀 Scaling Dataset"):
                X_train_scaled, X_test_scaled = scaling_data(
                    st.session_state.df, st.session_state.X_train, st.session_state.X_test
                )
                st.session_state.X_train = X_train_scaled
                st.session_state.X_test = X_test_scaled
                st.session_state.scaled = True
                st.success("Dataset berhasil di-scaling!")

            # Boxplot selalu ditampilkan (berubah tergantung sudah diskalakan atau belum)
            st.write("**Boxplot fitur numerikal**")
            col1, col2 = st.columns([1, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 4))

                # Pilih data sesuai status scaling
                if st.session_state.scaled:
                    boxplot_data = st.session_state.X_train.copy()
                    label_caption = "🔍 Menampilkan boxplot SETELAH scaling"
                    if "SeniorCitizen" in boxplot_data.columns:
                        boxplot_data = boxplot_data.drop(columns=["SeniorCitizen"])
                else:
                    boxplot_data = st.session_state.X_train.copy()
                    if "SeniorCitizen" in boxplot_data.columns:
                        boxplot_data = boxplot_data.drop(columns=["SeniorCitizen"])
                    label_caption = "🔍 Menampilkan boxplot SEBELUM scaling"

                sns.boxplot(data=boxplot_data, ax=ax)

                # Styling transparan dan warna putih
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.title.set_color("white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.tick_params(axis="x", colors="white")
                ax.tick_params(axis="y", colors="white")
                for spine in ax.spines.values():
                    spine.set_color("white")

                st.pyplot(fig)

            st.caption(label_caption)
                # st.subheader("🔹 Encoding Data")
                # st.write("Preview DataFrame Sebelum Encoding")
                # st.dataframe(df)

        # if st.button("🧩 Lakukan Encoding Data"):
        #         st.write("Preview DataFrame Setelah Encoding")
        #         encoded_df = encoding_data(df)
        #         st.session_state.encoded_df = encoded_df
        #         st.session_state.encoded_done = True
        #         st.success("Encoding berhasil!")


#         if st.session_state.convert_done and st.session_state.encoded_done:
            
#             encoded_df = st.session_state.encoded_df
#             st.write(f"Shape sebelum encoding: {df.shape}")
#             st.write(f"Shape setelah encoding: {encoded_df.shape}")

#         if "encoded_df" in st.session_state:
#             st.subheader("🔹 Distribusi Churn")

#             if "smote_df" not in st.session_state:
#                 st.session_state.smote_df = None

#             if st.button("📚 Sampling dengan SMOTE"):
#                 X_res, y_res = sampling_data(st.session_state.encoded_df)
#                 df_resampled = pd.DataFrame(X_res, columns=X_res.columns)
#                 df_resampled["Churn"] = y_res
#                 st.session_state.smote_df = df_resampled
#                 st.success("SMOTE sampling berhasil!")

#             if st.session_state.smote_df is not None:
#                 churn_counts = st.session_state.smote_df["Churn"].value_counts()
#                 st.info("Menampilkan distribusi setelah SMOTE.")
#             else:
#                 churn_counts = st.session_state.encoded_df["Churn"].value_counts()
#                 st.info("Menampilkan distribusi data asli.")

#             churn_counts.index = churn_counts.index.map({0: "No", 1: "Yes"})
#             st.bar_chart(churn_counts)
#         if st.session_state.get("smote_df") is not None:
#             st.subheader("🔹 Split & Scaling Data")
#             if st.button("✂️ Split & Scaling Data"):
#                 X = st.session_state.smote_df.drop("Churn", axis=1)
#                 y = st.session_state.smote_df["Churn"]
#                 X_train, X_test, y_train, y_test = splitting_scaling(X, y)

#                 st.session_state.X_train = X_train
#                 st.session_state.X_test = X_test
#                 st.session_state.y_train = y_train
#                 st.session_state.y_test = y_test

#                 st.success("Data berhasil di-split dan di-scaling!")
#                 st.write(f"Jumlah data training: {len(X_train)}")
#                 st.write(f"Jumlah data testing: {len(X_test)}")
#                 st.write(f"Proporsi: {len(X_train)/(len(X_train)+len(X_test)):.2%} training / {len(X_test)/(len(X_train)+len(X_test)):.2%} testing")
        

# with tab3:
#     st.header("Hyperparameter Tuning & Build Model")
    
#     if not all(k in st.session_state for k in ("X_train", "X_test", "y_train", "y_test")):
#         st.warning("⚠️ Anda perlu melakukan preprocessing, SMOTE, dan split & scaling pada tab sebelumnya terlebih dahulu.")
#         st.stop()
#     st.write("**Skema model yang akan diuji melalui Hyperparameter Tuning:**")
#     st.code("""classifiers = {
#                 'rf': RandomForestClassifier(random_state=42),
#                 'svm': SVC(probability=True, random_state=42),
#                 'lr': LogisticRegression(max_iter=1000, random_state=42)
#             }""", language="python")
#     st.write("**Parameter untuk setiap Skema Model:**")
#     st.code("""Random Forest:
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5,10],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True,False]
# Support Vector Machine:
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf'],
#     'gamma': ['scale', 'auto']
# Logistic Regression:
#     'C': [0.01, 0.1, 1, 10],
#     'penalty': ['l2'],
#     'solver': ['lbfgs']
# """,
#                 language="python"
#     )

#     gs = joblib.load("models/grid_search.pkl")
#     best_pipeline = joblib.load('models/best_model.pkl')

#     if isinstance(best_pipeline.named_steps['clf'], RandomForestClassifier):
#         model = best_pipeline.named_steps['clf']

#     st.write("**Skema model & Kombinasi Parameter Terbaik (Best Model & Parameters):**")
#     st.write(gs.best_params_)
#     st.write("**Skor Rata-Rata Cross Validation Terbaik (Best Score):**")
#     st.write(f"{gs.cv_results_['mean_test_score'][gs.best_index_]:.4f}")
#     st.write("**Standar Deviasi Cross Validation:**")
#     st.write(f"{gs.cv_results_['std_test_score'][gs.best_index_]:.4f}")


#     y_train_pred = model.predict(st.session_state.X_train)
#     y_pred = model.predict(st.session_state.X_test)

#     st.session_state.y_train_pred = y_train_pred
#     st.session_state.y_pred = y_pred
#     st.subheader("🔹 Training Model")
#     if st.button("Train Model"):
#         st.session_state.show_accuracy = True

#     if st.session_state.get("show_accuracy", False):
#         st.write(f"**Train Accuracy:** {accuracy_score(st.session_state.y_train, y_train_pred):.4f}")
#         st.write(f"**Test Accuracy:**  {accuracy_score(st.session_state.y_test, y_pred):.4f}")
        
#         st.subheader("🔹 Classification Report (Test Data)")
#         report_test_text = classification_report(st.session_state.y_test, y_pred)
#         st.markdown(f"```\n{report_test_text}\n```")

#         cm = confusion_matrix(st.session_state.y_test, y_pred)


#         importances = model.feature_importances_
#         features = st.session_state.X_train.columns
#         importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
#         st.session_state.importance_df = importance_df
#         st.subheader(" 🔹 Grafik Feature Importance (Fitur Lengkap)")
#         st.bar_chart(st.session_state.importance_df.set_index("Feature")["Importance"])

#         st.subheader("🔹Evaluasi Threshold untuk Feature Selection")

#         if st.button("Evaluasi Threshold Feature Importance"):
#             df_results = evaluate_thresholds(
#                 st.session_state.X_train, st.session_state.y_train,
#                 st.session_state.X_test, st.session_state.y_test,
#                 st.session_state.importance_df
#             )
#             st.session_state.threshold_eval_df = df_results
#             st.success("Evaluasi threshold selesai!")

#         if "threshold_eval_df" in st.session_state:
#             st.write("Hasil Evaluasi Threshold:")
#             st.dataframe(st.session_state.threshold_eval_df)

#             st.write("Grafik Threshold dan Akurasi (CV & Test)")
#             col1, col2 = st.columns([1, 1])
            
#             with col1:
#                 fig, ax = plt.subplots(figsize=(10, 6), facecolor='none') 
#                 ax.patch.set_alpha(0)

#                 ax.errorbar(
#                     st.session_state.threshold_eval_df['Threshold'], 
#                     st.session_state.threshold_eval_df['CV_Accuracy_Mean'],
#                     yerr=st.session_state.threshold_eval_df['CV_Accuracy_Std'], 
#                     fmt='-o', capsize=5, label='CV Accuracy'
#                 )
#                 ax.plot(
#                     st.session_state.threshold_eval_df['Threshold'], 
#                     st.session_state.threshold_eval_df['Test_Accuracy'], 
#                     '-s', label='Test Accuracy', color='orange'
#                 )

#                 ax.set_xlabel("Threshold", color='white')
#                 ax.set_ylabel("Accuracy", color='white')
#                 ax.set_title("Threshold vs Accuracy (CV & Test)", color='white')

#                 ax.tick_params(axis='x', colors='white')
#                 ax.tick_params(axis='y', colors='white')

#                 ax.grid(True, color='gray', alpha=0.3)
#                 legend = ax.legend(frameon=False)
#                 for text in legend.get_texts():
#                     text.set_color('white')

#                 st.pyplot(fig)

#         if "threshold_eval_df" in st.session_state and st.button("Analyze Important Features (Threshold 0.005)"):
#             threshold = 0.005
#             important_features = st.session_state.importance_df[
#                 st.session_state.importance_df['Importance'] >= threshold
#             ]
#             X_train_selected = st.session_state.X_train[important_features["Feature"]]
#             X_test_selected = st.session_state.X_test[important_features["Feature"]]

#             st.session_state.important_features = important_features
#             st.session_state.X_train_selected = X_train_selected
#             st.session_state.X_test_selected = X_test_selected
#             st.session_state.show_selected_features = True

#         if st.session_state.get("show_selected_features", False):
#             st.subheader("🔹Grafik Feature Importance dengan Threshold 0.005")
#             st.bar_chart(st.session_state.important_features.set_index("Feature")["Importance"])
#             st.write(f"Fitur yang dihapus: {len(st.session_state.X_train.columns) - len(st.session_state.important_features)}")
#             st.write(f"Jumlah fitur dengan importance >= 0.005: {len(st.session_state.important_features)}")
#             original_count = st.session_state.X_train.shape[1]
#             selected_count = len(st.session_state.important_features)
#             reduction_percent = ((original_count - selected_count) / original_count) * 100
#             st.write(f"📉 **Feature Reduction:** {original_count} → {selected_count} features ({reduction_percent:.1f}% reduction)")

#             st.subheader("🔹Selected Features")
#             st.dataframe(st.session_state.important_features.reset_index(drop=True))
            
#             st.subheader("🔹Retrain Model")
#             if "important_features" in st.session_state and st.button("Retrain Model after Feature Selection"):
#                 model_selected, selected_features = joblib.load("models/best_model_selected_features.pkl")
#                 X_train_selected = st.session_state.X_train[selected_features]
#                 X_test_selected = st.session_state.X_test[selected_features]
#                 y_train_pred_selected = model_selected.predict(X_train_selected)
#                 y_pred_selected = model_selected.predict(X_test_selected)

#                 st.session_state.retrained_model = {
#                     "model": model_selected,
#                     "y_train_pred": y_train_pred_selected,
#                     "y_pred": y_pred_selected,
#                     "selected_features": selected_features
#                 }
#                 st.session_state.show_retrain_result = True

#             if st.session_state.get("show_retrain_result", False):
#                 retrain = st.session_state.retrained_model
#                 if st.session_state.get("show_retrain_result", False):
#                     retrain = st.session_state.retrained_model

#                     train_acc = accuracy_score(st.session_state.y_train, retrain["y_train_pred"])
#                     test_acc = accuracy_score(st.session_state.y_test, retrain["y_pred"])
                    
#                     st.write(f"**Train Accuracy:** {train_acc:.4f}")
#                     st.write(f"**Test Accuracy:** {test_acc:.4f}")

                
#                     st.subheader(" 🔹Classification Report")
#                     report_test_text = classification_report(
#                         st.session_state.y_test, retrain["y_pred"]
#                     )
#                     st.markdown(f"```\n{report_test_text}\n```")

#                     cm_selected = confusion_matrix(st.session_state.y_test, retrain["y_pred"])
#                     st.subheader(" 🔹**Confusion Matrix sebelum & setelah Feature Selection**")
#                     col1, col2 = st.columns([1, 1])
                    
#                     with col1:
#                         fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
#                         sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,cbar=True)
#                         ax.set_title("Confusion Matrix\n(Before Feature Selection)", color = 'white')
#                         ax.set_xlabel("Predicted Label", color = "white")
#                         ax.set_ylabel("True Label", color = "white")
#                         st.pyplot(fig)
                        
#                     with col2:
#                         fig, ax = plt.subplots(figsize=(7, 5), facecolor='none')
#                         sns.heatmap(cm_selected, annot=True, fmt="d", cmap="Greens", ax=ax,cbar=True)
#                         ax.set_title("Confusion Matrix\n(After Feature Selection)", color = 'white')
#                         ax.set_xlabel("Predicted Label", color = "white")
#                         ax.set_ylabel("True Label", color = "white")
#                         st.pyplot(fig)
                        
