import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, ConfusionMatrixDisplay

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

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
}

print("--- MULTI-MODEL EVALUATION RESULTS ---")
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    trained_models[name] = model
    print(f"{name}: Accuracy = {acc*100:.2f}% | Recall = {rec*100:.2f}%")

rf_model = trained_models["Random Forest"]

fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test_scaled, y_test, ax=ax, cmap="Blues")
plt.title("Confusion Matrix - Random Forest Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=rf_model.feature_importances_, y=X.columns, palette="viridis")
plt.title("Feature Importances in Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Clinical Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Evaluation plots saved: confusion_matrix.png, feature_importance.png")
