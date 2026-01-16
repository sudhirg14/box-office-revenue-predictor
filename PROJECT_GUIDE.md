# 🎬 Movie Box Office Predictor - Complete Implementation Guide

## 🚀 Quick Start

### 1. Run the Application
```bash
# Method 1: Using Python directly
python app.py

# Method 2: Using the batch file (Windows)
run_app.bat

# Method 3: Using the demo script to test API
python demo.py
```

### 2. Access the Web Interface
- Open your browser and go to: **http://localhost:5000**
- Fill in the movie parameters
- Click "Predict Box Office Collection"

## 📊 Model Performance

Our trained model achieves excellent performance:
- **R² Score**: 0.7122 (71.22% variance explained)
- **RMSE**: 81.10 million dollars
- **MAE**: 64.75 million dollars
- **Cross-validation R²**: 0.7220 ± 0.0356

## 🏗️ Project Structure

```
movie-box-office-predictor/
├── 📁 data/                          # Dataset files
│   └── movie_box_office_uniform_5000.csv
├── 📁 models/                        # Trained ML models
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   ├── feature_columns.pkl          # Feature column names
│   ├── training_results.pkl         # Training metrics
│   └── feature_importance.csv       # Feature importance scores
├── 📁 static/                       # Web assets
│   ├── 📁 css/
│   │   └── style.css               # Custom styling
│   ├── 📁 js/
│   │   └── script.js               # Frontend JavaScript
│   └── 📁 images/                  # Generated charts
├── 📁 templates/                    # HTML templates
│   └── index.html                  # Main web interface
├── 📁 backend/                      # Backend code (future expansion)
├── 📁 frontend/                     # Frontend code (future expansion)
├── 📁 notebooks/                    # Jupyter notebooks for analysis
├── 📁 tests/                        # Unit tests
├── 📁 docs/                         # Documentation
├── 🐍 app.py                       # Main Flask application
├── 🐍 train_model.py               # ML model training script
├── 🐍 demo.py                      # API testing script
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                    # Project overview
├── 📄 PROJECT_GUIDE.md             # This guide
└── 🔧 run_app.bat                  # Windows batch file to run app
```

## 🧠 Machine Learning Details

### Dataset Information
- **Size**: 5,000 movies
- **Features**: 14 original features → 18 after preprocessing
- **Time Range**: 2000-2024
- **Genres**: 8 balanced categories

### Feature Engineering
1. **Categorical Encoding**: One-hot encoding for genres
2. **Feature Scaling**: StandardScaler for numerical features
3. **Feature Selection**: All features retained (no significant redundancy)

### Model Selection
We compared multiple algorithms:
- **Random Forest**: ✅ **Selected** (R² = 0.7122)
- **XGBoost**: R² = 0.7049

### Top 10 Most Important Features
1. **Budget** (63.00%) - Most critical factor
2. **Review Sentiment** (16.19%) - Audience perception
3. **Star Power** (5.66%) - Celebrity influence
4. **Runtime** (2.09%) - Movie length
5. **Social Media Buzz** (2.09%) - Online presence
6. **Marketing Spend** (2.07%) - Promotional budget
7. **Review Volume** (2.04%) - Number of reviews
8. **Critic Rating** (2.02%) - Professional reviews
9. **Audience Rating** (1.95%) - Public opinion
10. **Release Year** (1.67%) - Temporal factor

## 🌐 API Endpoints

### 1. Main Interface
- **GET** `/` - Web interface for predictions

### 2. Prediction API
- **POST** `/predict` - Get box office prediction
```json
{
  "genre": "Action",
  "budget_million": 150.0,
  "release_year": 2024,
  "runtime_min": 130,
  "critic_rating": 7.5,
  "audience_rating": 8.2,
  "review_sentiment": 0.8,
  "review_volume": 25000,
  "star_power": 0.9,
  "social_media_buzz": 200000,
  "marketing_spend_million": 50.0
}
```

### 3. Statistics API
- **GET** `/api/stats` - Dataset statistics
- **GET** `/api/features` - Feature importance

## 🎯 Usage Examples

### Example 1: High-Budget Blockbuster
```python
import requests

data = {
    "genre": "Action",
    "budget_million": 200.0,
    "release_year": 2024,
    "runtime_min": 140,
    "critic_rating": 8.0,
    "audience_rating": 8.5,
    "review_sentiment": 0.9,
    "review_volume": 50000,
    "star_power": 1.0,
    "social_media_buzz": 500000,
    "marketing_spend_million": 80.0
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
print(f"Predicted Box Office: ${result['prediction']:.2f}M")
```

### Example 2: Indie Film
```python
data = {
    "genre": "Drama",
    "budget_million": 3.0,
    "release_year": 2024,
    "runtime_min": 85,
    "critic_rating": 9.0,
    "audience_rating": 7.5,
    "review_sentiment": 0.6,
    "review_volume": 2000,
    "star_power": 0.1,
    "social_media_buzz": 5000,
    "marketing_spend_million": 1.0
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
print(f"Predicted Box Office: ${result['prediction']:.2f}M")
```

## 🔧 Development

### Training a New Model
```bash
python train_model.py
```

### Testing the API
```bash
python demo.py
```

### Adding New Features
1. Modify `train_model.py` to include new features
2. Update `app.py` to handle new input parameters
3. Update the web form in `templates/index.html`
4. Retrain the model

## 📈 Model Insights

### Key Findings
1. **Budget is King**: 63% of prediction comes from budget alone
2. **Sentiment Matters**: Review sentiment is the second most important factor
3. **Star Power Helps**: Celebrity influence significantly impacts box office
4. **Genre Effects**: Different genres have varying baseline expectations
5. **Marketing ROI**: Marketing spend has moderate impact

### Prediction Accuracy
- **High Accuracy**: For movies with budget > $50M (RMSE: ~$60M)
- **Moderate Accuracy**: For indie films < $10M (RMSE: ~$25M)
- **Confidence Range**: 60-95% based on feature alignment

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment
1. **Heroku**: Add `Procfile` and `runtime.txt`
2. **AWS EC2**: Use `gunicorn` for production server
3. **Docker**: Create `Dockerfile` for containerization

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🧪 Testing

### Manual Testing
1. Run the web interface
2. Test different movie scenarios
3. Verify prediction ranges make sense

### Automated Testing
```bash
python demo.py  # Tests all API endpoints
```

## 📝 Future Enhancements

### Phase 2 Features
1. **Advanced Models**: Neural networks, ensemble methods
2. **Real-time Data**: Integration with movie databases
3. **User Accounts**: Save prediction history
4. **Comparison Tool**: Compare multiple movies
5. **ROI Calculator**: Marketing spend optimization

### Phase 3 Features
1. **Mobile App**: React Native or Flutter
2. **API Authentication**: Secure endpoints
3. **Batch Predictions**: Multiple movies at once
4. **Export Features**: CSV/PDF reports
5. **Advanced Analytics**: Trend analysis, market insights

## 🐛 Troubleshooting

### Common Issues

1. **Port 5000 in use**
   ```bash
   # Use different port
   python app.py --port 5001
   ```

2. **Model not found**
   ```bash
   # Retrain the model
   python train_model.py
   ```

3. **Dependencies missing**
   ```bash
   # Install requirements
   pip install -r requirements.txt
   ```

4. **Prediction errors**
   - Check input ranges (1-10 for ratings, 0-1 for star power)
   - Ensure all required fields are filled
   - Verify data types (numbers, not strings)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the error messages in the console
3. Verify all dependencies are installed
4. Ensure the model files exist in the `models/` directory

---

**🎬 Happy Predicting! 🎬**

This system provides a solid foundation for movie box office prediction with room for significant enhancement and customization based on your specific needs.
