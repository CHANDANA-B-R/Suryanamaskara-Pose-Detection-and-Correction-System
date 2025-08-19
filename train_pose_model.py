# train_pose_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1️⃣ Load CSV
csv_path = "pose_landmarks.csv"  # Update path if needed
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} samples from {csv_path}")
if len(df) == 0:
    raise ValueError("CSV file is empty! Ensure landmark extraction worked correctly.")

# 2️⃣ Separate features & labels
X = df.drop("label", axis=1)
y = df["label"]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 4️⃣ RandomForest with Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# 5️⃣ Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6️⃣ Save model
joblib.dump(best_model, "pose_classifier_rf.pkl")
print("💾 Model saved as pose_classifier_rf.pkl")
