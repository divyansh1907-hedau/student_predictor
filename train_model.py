import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Load Dataset
df = pd.read_csv('students_uci.csv')

# 2. Map Features
data = pd.DataFrame()
data['attendance'] = np.clip(100 - (df['absences'] * 2.5), 0, 100)
data['midterm'] = (df['G1'] / 20.0) * 100
data['assignment'] = (df['G2'] / 20.0) * 100
data['logins'] = df['studytime'] * 12
data['study_hours'] = df['studytime'] * 5

# Target: 1 if final grade < 10 (At-Risk), else 0
data['at_risk'] = (df['G3'] < 10).astype(int)

X = data[['attendance', 'midterm', 'assignment', 'logins', 'study_hours']]
y = data['at_risk']

# 3. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models (Using Random Forest with probability calibration for smooth outputs)
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
}

metrics_summary = {}

for name, model in models.items():
  model.fit(X_train_scaled, y_train)
  y_pred = model.predict(X_test_scaled)
  y_proba = model.predict_proba(X_test_scaled)[:, 1]

  metrics_summary[name] = {
      'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
      'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
      'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
      'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
      'roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4),
  }

# Select Random Forest for smooth continuous risk probabilities (0%-100%)
best_model = models['Random Forest']

# Save Artifacts
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

with open('metrics.json', 'w') as f:
  json.dump(metrics_summary, f, indent=4)

print(
    'Retrained successfully with Random Forest Probability Estimator! Saved'
    ' model.pkl, scaler.pkl, and metrics.json.'
)