import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, mean_absolute_error, mean_squared_error)

# Algorithms listed in PPT slides 11 & 13
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('students.csv')

X = df[['attendance', 'midterm', 'assignment', 'logins', 'study_hours']]
y = df['at_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Define Algorithms specified in PPT
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# 3. Evaluate models on all PPT metrics (Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE)
results = []
best_model = None
best_auc = -1
best_model_name = ""

print("\n--- MODEL EVALUATION MATRIX ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except Exception:
        auc = 0.5
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    results.append({
        "Model": name, "Accuracy": f"{acc:.2f}", "Precision": f"{prec:.2f}",
        "Recall": f"{rec:.2f}", "F1-Score": f"{f1:.2f}", "ROC-AUC": f"{auc:.2f}",
        "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}"
    })
    
    # Save the highest performing model based on ROC-AUC
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# Print comparison table in terminal
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print(f"\n🏆 Best Performing Model: {best_model_name} (ROC-AUC: {best_auc:.2f})")

# Save the best model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Saved best model to 'model.pkl' successfully!")