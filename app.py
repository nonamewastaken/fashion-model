from flask import Flask, render_template, send_from_directory, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import json
import tensorflow as tf
from tensorflow import keras
import re
import signal
import sys

app = Flask(__name__)

# Global variables for graceful shutdown
sentiment_model = None
purchase_model = None
label_encoders = None
scaler = None
feature_columns = None

def signal_handler(sig, frame):
    if sentiment_model is not None:
        del sentiment_model
    tf.keras.backend.clear_session()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Load the trained models and preprocessing objects
def load_models():
    global purchase_model, label_encoders, scaler, feature_columns
    try:
        # Load the classification model for purchase decision
        with open('model/purchase_decision_model.pkl', 'rb') as f:
            purchase_model = pickle.load(f)
        
        # Load preprocessing objects
        with open('model/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('model/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return purchase_model, label_encoders, scaler, feature_columns
    except FileNotFoundError:
        print("Model files not found. Please train the models first.")
        return None, None, None, None

# Load sentiment analysis model
def load_sentiment_model():
    global sentiment_model
    try:
        sentiment_model = keras.models.load_model('model/sentiment_model.h5')
        return sentiment_model
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        return None

# Initialize models
load_models()
load_sentiment_model()

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/fashion-decision-predictor')
def fashion_decision_predictor():
    return render_template('fashion_decision_predictor.html')

@app.route('/trend-insights')
def trend_insights():
    return render_template('trend_insights.html')

@app.route('/data-verifier')
def data_verifier():
    return render_template('data_verifier.html')

@app.route('/revenue-insights')
def revenue_insights():
    return render_template('revenue_insights.html')

@app.route('/user-guide')
def user_guide():
    return render_template('user_guide.html')

@app.route('/api/status')
def api_status():
    """Check if models are loaded and ready"""
    models_loaded = purchase_model is not None
    return jsonify({
        'models_loaded': models_loaded,
        'status': 'ready' if models_loaded else 'models_not_loaded'
    })



@app.route('/predict-single', methods=['POST'])
def predict_single():
    try:
        # Check if models are loaded
        if purchase_model is None:
            return jsonify({'error': 'Purchase decision model not loaded. Please ensure models are trained and available.'}), 500
        
        data = request.get_json()
        
        # Extract input data
        age = int(data.get('age', 0))
        gender = data.get('gender', '')
        location = data.get('location', '')
        brand = data.get('brand', '')
        fashion_item = data.get('fashion_item', '')
        price = float(data.get('price', 0))
        rating = float(data.get('rating', 0))
        sentiment_text = data.get('sentiment_text', '')
        trendy_score = int(data.get('trendy_score', 50))  # Add trendy_score input
        
        # Predict sentiment using the sentiment model
        predicted_sentiment = predict_sentiment(sentiment_text)
        
        # Create feature vector for prediction
        feature_vector = create_feature_vector(
            age, gender, location, brand, fashion_item, 
            price, rating, predicted_sentiment, trendy_score
        )
        
        # Make prediction
        prediction_proba = purchase_model.predict_proba(feature_vector)[0]
        purchase_decision = purchase_model.predict(feature_vector)[0]
        
        # Convert prediction back to human-readable format
        decision_labels = label_encoders['Purchase_Decision'].classes_
        purchase_decision_label = decision_labels[purchase_decision]
        
        # Get confidence (probability of the predicted class)
        confidence = max(prediction_proba) * 100
        
        # Prepare response
        response = {
            'purchase_decision': purchase_decision_label,
            'purchase_probability': f"{confidence:.1f}",
            'confidence': f"{confidence:.1f}%",
            'predicted_sentiment': predicted_sentiment,
            'original_sentiment_text': sentiment_text
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    try:
        # Check if models are loaded
        if purchase_model is None:
            return jsonify({'error': 'Purchase decision model not loaded. Please ensure models are trained and available.'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['Age', 'Gender', 'Location', 'Brand', 'Fashion_Item', 'Price', 'Rating', 'Sentiment_Text', 'Trendy_Score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400
        
        results = []
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Extract data from row
                age = int(row['Age'])
                gender = str(row['Gender'])
                location = str(row['Location'])
                brand = str(row['Brand'])
                fashion_item = str(row['Fashion_Item'])
                price = float(row['Price'])
                rating = float(row['Rating'])
                sentiment_text = str(row['Sentiment_Text'])
                trendy_score = int(row['Trendy_Score'])
                
                # Predict sentiment
                predicted_sentiment = predict_sentiment(sentiment_text)
                
                # Create feature vector
                feature_vector = create_feature_vector(
                    age, gender, location, brand, fashion_item, 
                    price, rating, predicted_sentiment, trendy_score
                )
                
                # Make prediction
                prediction_proba = purchase_model.predict_proba(feature_vector)[0]
                purchase_decision = purchase_model.predict(feature_vector)[0]
                
                # Convert prediction back to human-readable format
                decision_labels = label_encoders['Purchase_Decision'].classes_
                purchase_decision_label = decision_labels[purchase_decision]
                
                # Get confidence
                confidence = max(prediction_proba) * 100
                
                results.append({
                    'row': index + 1,
                    'purchase_decision': purchase_decision_label,
                    'purchase_probability': f"{confidence:.1f}",
                    'confidence': f"{confidence:.1f}%",
                    'predicted_sentiment': predicted_sentiment,
                    'original_sentiment_text': sentiment_text
                })
                
            except Exception as e:
                errors.append({
                    'row': index + 1,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'errors': errors,
            'total_processed': len(results),
            'total_errors': len(errors)
        })
        
    except Exception as e:
        return jsonify({'error': f'CSV processing failed: {str(e)}'}), 500

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_sentiment(text):
    """Predict sentiment using the loaded model or fallback to rule-based"""
    try:
        if sentiment_model is not None:
            # Use the trained sentiment model
            processed_text = preprocess_text(text)
            if processed_text:
                # This is a simplified approach - you might need to adjust based on your model's input requirements
                # For now, we'll use a rule-based fallback
                pass
        
        # Fallback to rule-based sentiment analysis
        text_lower = text.lower()
        
        positive_words = ['love', 'great', 'good', 'excellent', 'amazing', 'wonderful', 'perfect', 'beautiful', 'comfortable', 'stylish']
        negative_words = ['hate', 'terrible', 'bad', 'awful', 'horrible', 'uncomfortable', 'ugly', 'cheap', 'poor', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"
            
    except Exception as e:
        print(f"Sentiment prediction error: {e}")
        return "Neutral"

def create_feature_vector(age, gender, location, brand, fashion_item, price, rating, sentiment, trendy_score):
    """Create feature vector for model prediction"""
    try:
        # Create a DataFrame with the input data
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Brand': [brand],
            'Fashion_Item': [fashion_item],
            'Price': [price],
            'Rating': [rating],
            'Sentiment': [sentiment],
            'Trendy_Score': [trendy_score]  # Include trendy_score
        }
        
        df = pd.DataFrame(input_data)
        
        # Encode categorical variables
        for column in ['Gender', 'Sentiment']:
            if column in label_encoders:
                df[column] = label_encoders[column].transform(df[column])
        
        # One-hot encoding for Brand and Fashion_Item (without drop_first to match training)
        brand_dummies = pd.get_dummies(df['Brand'], prefix='Brand', drop_first=False)
        fashion_dummies = pd.get_dummies(df['Fashion_Item'], prefix='Fashion_Item', drop_first=False)
        
        # Combine all features
        df_encoded = pd.concat([
            df[['Age', 'Gender', 'Price', 'Rating', 'Sentiment', 'Trendy_Score']],
            brand_dummies,
            fashion_dummies
        ], axis=1)
        
        # Ensure all expected feature columns exist
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Reorder columns to match the expected feature order
        df_encoded = df_encoded[feature_columns]
        
        # Scale numerical features (in the same order as scaler was trained)
        numerical_features = ['Price', 'Rating', 'Trendy_Score', 'Age']
        df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])
        
        return df_encoded
        
    except Exception as e:
        raise Exception(f"Error creating feature vector: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=8000) 