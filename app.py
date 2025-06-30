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
    try:
        data = request.get_json()
        print("Received data:", data)

        features = data.get('features')
        print("Features:", features)

        if features is None:
            return jsonify({'error': 'Missing "features" in request body'}), 400

        prediction = model.predict([features])
        print("Prediction:", prediction)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
