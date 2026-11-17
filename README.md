# Heart Disease Risk Prediction Using Machine Learning

An end-to-end machine learning prototype built using the UCI Cleveland Heart Disease Dataset (303 records, 13 predictors).

### Group Members:
- **Soubhagya Ranjan Behera** (1AM23CI148)
- **Shashank S** (1AM23CI139)
- **Shreyas R** (1AM23CI141)
- **Samrudh S** (1AM23CI127)

## Models Evaluated
- **Logistic Regression** (Linear Baseline)
- **Support Vector Machine (SVM)**
- **Random Forest Classifier** (`n_estimators=100`, `class_weight='balanced'`) — Selected for final deployment

## Project Files
- `train_and_evaluate.py`: Multi-model training script and evaluation plot generator.
- `app.py`: Interactive Streamlit application with multi-model metrics display.
- `requirements.txt`: Python package dependencies.
