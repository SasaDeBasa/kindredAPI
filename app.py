from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Health check for Railway
@app.route('/')
def health():
    return "Mental Health Model is running!", 200

# Load model
model = joblib.load('depression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run()
