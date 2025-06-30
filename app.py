from flask import Flask, request, jsonify
import joblib
import os

print("Current working directory:", os.getcwd())
print("Files in working directory:", os.listdir("."))

model = joblib.load('depression_model.pkl')

app = Flask(__name__)

# Health check for Railway
@app.route('/')
def health():
    return "Mental Health Model is running!", 200

# Load model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run()
