import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("ovarian_cancer_prediction_dataset.csv")

# Step 2: Encode categorical features
categorical_cols = ["Family_History", "BRCA_Mutation", "Hormone_Therapy",
                    "Endometriosis", "Infertility", "Obesity", "Smoking", "Ovarian_Cancer"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Split features and labels
X = df.drop("Ovarian_Cancer", axis=1)
y = df["Ovarian_Cancer"]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 7: Initialize ensemble classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Voting ensemble
ensemble_model = VotingClassifier(estimators=[
    ('random_forest', rf),
    ('gradient_boosting', gb)
], voting='soft')

# Step 8: Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = ensemble_model.predict(X_test)
y_proba = ensemble_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("🔍 Classification Report:\n", report)
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"📈 ROC-AUC Score: {roc_auc:.4f}")

# Step 10: Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Ovarian Cancer Prediction')
plt.legend(loc="lower right")
plt.grid()
plt.savefig("roc_curve.png")
plt.close()

# Step 11: Save the model and scaler
joblib.dump(ensemble_model, "ovarian_cancer_ensemble_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("📦 Model and preprocessing tools saved successfully.")
