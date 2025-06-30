from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("depression_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction[0]})
