import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title="CardioCheck ML", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")

@st.cache_resource
def train_models():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, names=cols, na_values="?")
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    df['target'] = (df['target'] > 0).astype(int)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
    svm = SVC(probability=True, random_state=42).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42).fit(X_train_scaled, y_train)

    metrics = {
        "Logistic Regression": (accuracy_score(y_test, lr.predict(X_test_scaled)), recall_score(y_test, lr.predict(X_test_scaled))),
        "SVM": (accuracy_score(y_test, svm.predict(X_test_scaled)), recall_score(y_test, svm.predict(X_test_scaled))),
        "Random Forest": (accuracy_score(y_test, rf.predict(X_test_scaled)), recall_score(y_test, rf.predict(X_test_scaled)))
    }
    return rf, scaler, metrics

model, scaler, metrics = train_models()

st.sidebar.header("Model Comparison (Test Set)")
for name, (acc, rec) in metrics.items():
    st.sidebar.markdown(f"**{name}**")
    st.sidebar.write(f"Accuracy: `{acc*100:.1f}%` | Recall: `{rec*100:.1f}%`")

st.sidebar.divider()
st.sidebar.info("Primary Deployed Model: **Random Forest Classifier**")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 85, 54)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 130)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 120, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])

with col2:
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox("Slope", [1, 2, 3], format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x])
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [3.0, 6.0, 7.0], format_func=lambda x: {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversible Defect"}[x])

if st.button("Run Diagnostic Prediction", use_container_width=True):
    raw_inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled = scaler.transform(raw_inputs)
    prob = model.predict_proba(scaled)[0][1] * 100
    if model.predict(scaled)[0] == 1:
        st.error(f"⚠️ High Risk Detected - Estimated Probability: {prob:.1f}%")
    else:
        st.success(f"✅ Low Risk / Healthy Profile - Probability: {prob:.1f}%")
