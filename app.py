from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('depression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

@app.route('/')
def home():
    return "Mental Health Model is running!"

if __name__ == '__main__':
    app.run()
