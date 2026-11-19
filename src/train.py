
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load Wine Dataset
wine = load_wine()

X = wine.data
y = wine.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("Wine Classification")
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "models/wine_model.pkl")

print("Model saved successfully!")
