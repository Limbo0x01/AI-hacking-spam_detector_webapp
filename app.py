from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model = joblib.load('spam_detector_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get message from the POST request
        message = request.form['message']
        
        if not message:
            return jsonify({'error': 'Please enter a message'})
        
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Transform the message using vectorizer
        message_vectorized = vectorizer.transform([processed_message])
        
        # Make prediction
        prediction = model.predict(message_vectorized)[0]
        probability = np.max(model.predict_proba(message_vectorized)) * 100
        
        # Determine result
        result = 'SPAM' if prediction == 1 else 'HAM'
        
        return jsonify({
            'message': message,
            'prediction': result,
            'probability': f'{probability:.2f}%'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)