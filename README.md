# Movie Box Office Prediction System

A machine learning-powered web application that predicts movie box office collection based on various movie features.

## Features

- **Real-time Predictions**: Input movie parameters and get instant box office predictions
- **Multiple ML Models**: Uses Random Forest, XGBoost, and other algorithms
- **Interactive Dashboard**: Beautiful web interface with data visualizations
- **Feature Analysis**: Understand which factors most influence box office success
- **Historical Data**: View and analyze the training dataset

## Dataset

The model is trained on a dataset of 5,000 movies with the following features:
- **Financial**: Budget, marketing spend
- **Content**: Genre, runtime, release year
- **Quality Metrics**: Critic rating, audience rating, review sentiment
- **Marketing**: Social media buzz, review volume, star power

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Web Interface**:
   Open your browser and go to `http://localhost:5000`

## Project Structure

```
├── backend/           # Flask API backend
├── frontend/          # HTML/CSS/JS frontend
├── models/           # Trained ML models
├── data/             # Dataset files
├── notebooks/        # Jupyter notebooks for analysis
├── static/           # Static web assets
├── templates/        # HTML templates
└── tests/            # Unit tests
```

## API Endpoints

- `GET /` - Main prediction interface
- `POST /predict` - Get box office prediction
- `GET /api/stats` - Dataset statistics
- `GET /api/features` - Feature importance

## Model Performance

The trained models achieve:
- **RMSE**: ~45M (Root Mean Square Error)
- **R² Score**: ~0.85
- **MAE**: ~35M (Mean Absolute Error)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

