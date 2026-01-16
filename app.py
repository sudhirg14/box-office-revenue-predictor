from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables for model and data
model = None
feature_columns = None
scaler = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, feature_columns, scaler
    
    try:
        model = joblib.load('models/best_model.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return False
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get input data
        data = request.get_json()
        
        # Prepare feature vector
        features = prepare_features(data)
        
        # Make prediction
        prediction = model.predict(features.reshape(1, -1))[0]
        
        # Get prediction confidence (simplified)
        confidence = min(95, max(60, 100 - abs(prediction - 313) / 10))
        
        return jsonify({
            'prediction': round(prediction, 2),
            'confidence': round(confidence, 1),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    try:
        df = pd.read_csv('data/movie_box_office_uniform_5000.csv')
        
        stats = {
            'total_movies': len(df),
            'avg_box_office': round(df['box_office_collection'].mean(), 2),
            'max_box_office': round(df['box_office_collection'].max(), 2),
            'min_box_office': round(df['box_office_collection'].min(), 2),
            'genres': df['genre'].value_counts().to_dict(),
            'year_range': {
                'min': int(df['release_year'].min()),
                'max': int(df['release_year'].max())
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features')
def get_feature_importance():
    """Get feature importance from the model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            features = feature_columns
            
            feature_importance = dict(zip(features, importance))
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return jsonify(dict(sorted_features))
        else:
            return jsonify({'error': 'Feature importance not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def prepare_features(data):
    """Prepare features for prediction"""
    # Create feature vector in the same order as training
    features = np.zeros(len(feature_columns))
    
    # Map input data to feature columns
    feature_mapping = {
        'budget_million': 'budget_million',
        'release_year': 'release_year',
        'runtime_min': 'runtime_min',
        'critic_rating': 'critic_rating',
        'audience_rating': 'audience_rating',
        'review_sentiment': 'review_sentiment',
        'review_volume': 'review_volume',
        'star_power': 'star_power',
        'social_media_buzz': 'social_media_buzz',
        'marketing_spend_million': 'marketing_spend_million'
    }
    
    for input_key, feature_name in feature_mapping.items():
        if input_key in data and feature_name in feature_columns:
            idx = feature_columns.index(feature_name)
            features[idx] = float(data[input_key])
    
    # Handle genre (one-hot encoding)
    genre = data.get('genre', 'Action')
    genre_features = [col for col in feature_columns if col.startswith('genre_')]
    for genre_feature in genre_features:
        if genre_feature == f'genre_{genre}':
            idx = feature_columns.index(genre_feature)
            features[idx] = 1
    
    # Apply scaling
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1)).flatten()
    
    return features

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)

