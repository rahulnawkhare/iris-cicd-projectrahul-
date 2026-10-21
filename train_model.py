import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained successfully!")
print("Model saved as iris_model.pkl")
