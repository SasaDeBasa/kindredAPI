from flask import Flask, request, jsonify
import joblib

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
    features = data.get('features')
    if not features or not isinstance(features, list):
        return jsonify({'error': 'Invalid input. Expected JSON with a "features" list.'}), 400

    try:
        prediction = model.predict([features])
        result = prediction[0]
        return jsonify({'prediction': int(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run()
