
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")

flower_names = ["setosa", "versicolor", "virginica"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris ML REST API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send valid JSON"}), 400

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    try:
        values = [float(data[key]) for key in features]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide all four numeric measurements"
        }), 400

    if any(value <= 0 or value > 10 for value in values):
        return jsonify({
            "error": "Measurements must be between 0 and 10 cm"
        }), 400

    prediction = int(model.predict([values])[0])

    return jsonify({
        "prediction": flower_names[prediction]
    })


#if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

