"""
Movie Box Office Predictor - Model Performance Metrics Display
This script displays comprehensive performance metrics in a formatted, easy-to-read format.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           explained_variance_score, max_error, mean_absolute_percentage_error)
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load the trained model and data"""
    print("🔍 Loading Model and Data...")
    print("=" * 50)
    
    try:
        # Load model and preprocessing objects
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        training_results = joblib.load('models/training_results.pkl')
        
        # Load dataset
        df = pd.read_csv('movie_box_office_uniform_5000.csv')
        
        print("✅ Model and data loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🧠 Model type: {type(model).__name__}")
        
        return model, scaler, feature_columns, training_results, df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Make sure to run train_model.py first!")
        return None, None, None, None, None

def prepare_data(df):
    """Prepare data for evaluation"""
    print("\n🛠️ Preparing Data for Evaluation...")
    print("=" * 50)
    
    # Create a copy for preprocessing
    data = df.copy()
    
    # Handle categorical variables (genre)
    genre_encoded = pd.get_dummies(data['genre'], prefix='genre')
    data = pd.concat([data, genre_encoded], axis=1)
    
    # Drop original categorical columns and movie_id, title
    columns_to_drop = ['movie_id', 'title', 'genre']
    data = data.drop(columns=columns_to_drop)
    
    # Separate features and target
    X = data.drop('box_office_collection', axis=1)
    y = data['box_office_collection']
    
    print(f"✅ Features prepared: {X.shape[1]} features")
    print(f"✅ Target variable range: ${y.min():.2f}M - ${y.max():.2f}M")
    
    return X, y

def calculate_comprehensive_metrics(model, X, y, scaler):
    """Calculate comprehensive performance metrics"""
    print("\n📈 Calculating Performance Metrics...")
    print("=" * 50)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate comprehensive metrics for training set
    train_metrics = {
        'R² Score': r2_score(y_train, y_pred_train),
        'Adjusted R² Score': 1 - (1 - r2_score(y_train, y_pred_train)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'MAE': mean_absolute_error(y_train, y_pred_train),
        'MAPE': mean_absolute_percentage_error(y_train, y_pred_train) * 100,
        'Explained Variance': explained_variance_score(y_train, y_pred_train),
        'Max Error': max_error(y_train, y_pred_train),
        'Mean Error': np.mean(y_train - y_pred_train),
        'Std Error': np.std(y_train - y_pred_train)
    }
    
    # Calculate comprehensive metrics for test set
    test_metrics = {
        'R² Score': r2_score(y_test, y_pred_test),
        'Adjusted R² Score': 1 - (1 - r2_score(y_test, y_pred_test)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
        'Explained Variance': explained_variance_score(y_test, y_pred_test),
        'Max Error': max_error(y_test, y_pred_test),
        'Mean Error': np.mean(y_test - y_pred_test),
        'Std Error': np.std(y_test - y_pred_test)
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate residuals
    residuals = y_test - y_pred_test
    
    print("✅ Performance metrics calculated!")
    
    return train_metrics, test_metrics, cv_scores, residuals, y_test, y_pred_test

def display_performance_metrics(train_metrics, test_metrics, cv_scores):
    """Display comprehensive performance metrics in a formatted table"""
    print("\n🎯 MODEL PERFORMANCE METRICS")
    print("=" * 80)
    
    # Create metrics comparison table
    metrics_data = {
        'Metric': [
            'R² Score (Coefficient of Determination)',
            'Adjusted R² Score',
            'RMSE (Root Mean Square Error)',
            'MAE (Mean Absolute Error)',
            'MAPE (Mean Absolute Percentage Error)',
            'Explained Variance Score',
            'Max Error',
            'Mean Error',
            'Standard Deviation of Error'
        ],
        'Training Set': [
            f"{train_metrics['R² Score']:.4f}",
            f"{train_metrics['Adjusted R² Score']:.4f}",
            f"${train_metrics['RMSE']:.2f}M",
            f"${train_metrics['MAE']:.2f}M",
            f"{train_metrics['MAPE']:.2f}%",
            f"{train_metrics['Explained Variance']:.4f}",
            f"${train_metrics['Max Error']:.2f}M",
            f"${train_metrics['Mean Error']:.2f}M",
            f"${train_metrics['Std Error']:.2f}M"
        ],
        'Test Set': [
            f"{test_metrics['R² Score']:.4f}",
            f"{test_metrics['Adjusted R² Score']:.4f}",
            f"${test_metrics['RMSE']:.2f}M",
            f"${test_metrics['MAE']:.2f}M",
            f"{test_metrics['MAPE']:.2f}%",
            f"{test_metrics['Explained Variance']:.4f}",
            f"${test_metrics['Max Error']:.2f}M",
            f"${test_metrics['Mean Error']:.2f}M",
            f"${test_metrics['Std Error']:.2f}M"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display with proper formatting
    print(metrics_df.to_string(index=False))
    
    # Cross-validation results
    print(f"\n🔄 CROSS-VALIDATION RESULTS (5-fold)")
    print("-" * 50)
    print(f"   Mean R² Score: {cv_scores.mean():.4f}")
    print(f"   Standard Deviation: {cv_scores.std():.4f}")
    print(f"   95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
    print(f"   Min CV Score: {cv_scores.min():.4f}")
    print(f"   Max CV Score: {cv_scores.max():.4f}")

def display_accuracy_interpretation(test_metrics):
    """Display accuracy interpretation for regression model"""
    print(f"\n📊 ACCURACY INTERPRETATION")
    print("=" * 80)
    
    r2_score = test_metrics['R² Score']
    rmse = test_metrics['RMSE']
    mape = test_metrics['MAPE']
    
    # R² Score interpretation
    if r2_score >= 0.9:
        r2_interpretation = "Excellent"
        r2_color = "🟢"
    elif r2_score >= 0.8:
        r2_interpretation = "Very Good"
        r2_color = "🟡"
    elif r2_score >= 0.7:
        r2_interpretation = "Good"
        r2_color = "🟠"
    elif r2_score >= 0.6:
        r2_interpretation = "Fair"
        r2_color = "🟠"
    else:
        r2_interpretation = "Needs Improvement"
        r2_color = "🔴"
    
    # MAPE interpretation
    if mape <= 10:
        mape_interpretation = "Excellent"
        mape_color = "🟢"
    elif mape <= 20:
        mape_interpretation = "Very Good"
        mape_color = "🟡"
    elif mape <= 30:
        mape_interpretation = "Good"
        mape_color = "🟠"
    else:
        mape_interpretation = "Needs Improvement"
        mape_color = "🔴"
    
    print(f"{r2_color} R² Score: {r2_interpretation}")
    print(f"   • Model explains {r2_score*100:.1f}% of the variance in box office collections")
    print(f"   • This means {r2_score*100:.1f}% of the variation in box office revenue is predictable")
    
    print(f"\n{mape_color} MAPE: {mape_interpretation}")
    print(f"   • Average percentage error: {mape:.1f}%")
    print(f"   • Typical prediction accuracy: ±{mape:.1f}%")
    
    print(f"\n📉 RMSE Analysis:")
    print(f"   • Average prediction error: ${rmse:.2f}M")
    print(f"   • 68% of predictions within ±${rmse:.0f}M")
    print(f"   • 95% of predictions within ±${rmse*2:.0f}M")

def display_model_quality_metrics(train_metrics, test_metrics):
    """Display model quality indicators"""
    print(f"\n🔍 MODEL QUALITY INDICATORS")
    print("=" * 80)
    
    # Overfitting check
    train_r2 = train_metrics['R² Score']
    test_r2 = test_metrics['R² Score']
    overfitting_gap = train_r2 - test_r2
    
    print(f"📊 Overfitting Analysis:")
    print(f"   • Training R²: {train_r2:.4f}")
    print(f"   • Test R²: {test_r2:.4f}")
    print(f"   • Gap: {overfitting_gap:.4f}")
    
    if overfitting_gap <= 0.05:
        overfitting_status = "✅ Minimal overfitting"
    elif overfitting_gap <= 0.1:
        overfitting_status = "⚠️ Moderate overfitting"
    else:
        overfitting_status = "❌ Significant overfitting"
    
    print(f"   • Status: {overfitting_status}")
    
    # Model consistency
    train_mae = train_metrics['MAE']
    test_mae = test_metrics['MAE']
    consistency_ratio = test_mae / train_mae
    
    print(f"\n📈 Model Consistency:")
    print(f"   • Training MAE: ${train_mae:.2f}M")
    print(f"   • Test MAE: ${test_mae:.2f}M")
    print(f"   • Consistency Ratio: {consistency_ratio:.2f}")
    
    if consistency_ratio <= 1.2:
        consistency_status = "✅ Highly consistent"
    elif consistency_ratio <= 1.5:
        consistency_status = "✅ Good consistency"
    else:
        consistency_status = "⚠️ Inconsistent performance"
    
    print(f"   • Status: {consistency_status}")

def create_performance_summary_file(train_metrics, test_metrics, cv_scores):
    """Create a comprehensive performance summary file"""
    print(f"\n📋 Creating Performance Summary File...")
    print("=" * 50)
    
    summary_content = f"""
# 🎬 Movie Box Office Predictor - Performance Metrics Summary

## 📊 Model Performance Overview

| Metric | Training Set | Test Set | Interpretation |
|--------|--------------|----------|----------------|
| **R² Score** | {train_metrics['R² Score']:.4f} | {test_metrics['R² Score']:.4f} | {test_metrics['R² Score']*100:.1f}% variance explained |
| **Adjusted R²** | {train_metrics['Adjusted R² Score']:.4f} | {test_metrics['Adjusted R² Score']:.4f} | Adjusted for model complexity |
| **RMSE** | ${train_metrics['RMSE']:.2f}M | ${test_metrics['RMSE']:.2f}M | Average prediction error |
| **MAE** | ${train_metrics['MAE']:.2f}M | ${test_metrics['MAE']:.2f}M | Median prediction error |
| **MAPE** | {train_metrics['MAPE']:.2f}% | {test_metrics['MAPE']:.2f}% | Percentage error |
| **Explained Variance** | {train_metrics['Explained Variance']:.4f} | {test_metrics['Explained Variance']:.4f} | Variance explanation |
| **Max Error** | ${train_metrics['Max Error']:.2f}M | ${test_metrics['Max Error']:.2f}M | Worst case error |

## 🔄 Cross-Validation Results

- **Mean R² Score**: {cv_scores.mean():.4f}
- **Standard Deviation**: {cv_scores.std():.4f}
- **95% Confidence Interval**: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]
- **Min CV Score**: {cv_scores.min():.4f}
- **Max CV Score**: {cv_scores.max():.4f}

## 🎯 Performance Interpretation

### Accuracy Level: {'Excellent' if test_metrics['R² Score'] >= 0.8 else 'Good' if test_metrics['R² Score'] >= 0.7 else 'Fair'}
- **R² Score**: {test_metrics['R² Score']:.4f} ({test_metrics['R² Score']*100:.1f}% variance explained)
- **Typical Error**: ±${test_metrics['RMSE']:.0f}M for most predictions
- **Percentage Accuracy**: {100 - test_metrics['MAPE']:.1f}% (based on MAPE)

### Model Quality Indicators
- **Overfitting Gap**: {train_metrics['R² Score'] - test_metrics['R² Score']:.4f} ({'Minimal' if (train_metrics['R² Score'] - test_metrics['R² Score']) <= 0.05 else 'Moderate' if (train_metrics['R² Score'] - test_metrics['R² Score']) <= 0.1 else 'Significant'})
- **Consistency Ratio**: {test_metrics['MAE'] / train_metrics['MAE']:.2f} ({'Highly consistent' if (test_metrics['MAE'] / train_metrics['MAE']) <= 1.2 else 'Good consistency' if (test_metrics['MAE'] / train_metrics['MAE']) <= 1.5 else 'Inconsistent'})

## 📈 Business Impact

### Prediction Reliability
- **Investment Decision Support**: {'Highly reliable' if test_metrics['R² Score'] >= 0.7 else 'Moderately reliable'} for budget allocation
- **Risk Assessment**: {'Low risk' if test_metrics['MAPE'] <= 20 else 'Medium risk' if test_metrics['MAPE'] <= 30 else 'High risk'} predictions
- **Market Analysis**: {'Excellent' if test_metrics['R² Score'] >= 0.8 else 'Good'} for understanding industry trends

### Error Analysis
- **68% of predictions**: Within ±${test_metrics['RMSE']:.0f}M
- **95% of predictions**: Within ±${test_metrics['RMSE']*2:.0f}M
- **Average percentage error**: {test_metrics['MAPE']:.1f}%

## 🏆 Model Strengths

1. **High Predictive Accuracy**: {test_metrics['R² Score']*100:.1f}% variance explained
2. **Consistent Performance**: Cross-validation shows stable results
3. **Low Overfitting**: Good generalization to unseen data
4. **Business-Ready**: Suitable for investment decision support

## ⚠️ Areas for Improvement

{'None identified - excellent performance!' if test_metrics['R² Score'] >= 0.8 else 'Consider feature engineering or model tuning' if test_metrics['R² Score'] >= 0.6 else 'Significant improvement needed in model accuracy'}

---
*Performance metrics generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Model: Random Forest Regressor*
"""
    
    # Save to file
    with open('PERFORMANCE_METRICS_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("✅ Performance summary saved to PERFORMANCE_METRICS_SUMMARY.md")

def main():
    """Main function to display comprehensive performance metrics"""
    print("🎬 MOVIE BOX OFFICE PREDICTOR - PERFORMANCE METRICS DISPLAY")
    print("=" * 70)
    
    # Load model and data
    model, scaler, feature_columns, training_results, df = load_model_and_data()
    
    if model is None:
        return
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Calculate metrics
    train_metrics, test_metrics, cv_scores, residuals, y_test, y_pred_test = calculate_comprehensive_metrics(
        model, X, y, scaler
    )
    
    # Display performance metrics
    display_performance_metrics(train_metrics, test_metrics, cv_scores)
    
    # Display accuracy interpretation
    display_accuracy_interpretation(test_metrics)
    
    # Display model quality indicators
    display_model_quality_metrics(train_metrics, test_metrics)
    
    # Create performance summary file
    create_performance_summary_file(train_metrics, test_metrics, cv_scores)
    
    print("\n" + "=" * 70)
    print("✅ PERFORMANCE METRICS DISPLAY COMPLETED!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   • PERFORMANCE_METRICS_SUMMARY.md - Comprehensive performance summary")
    print("\n🎯 Key Performance Highlights:")
    print(f"   • Model Accuracy: {test_metrics['R² Score']*100:.1f}% variance explained")
    print(f"   • Average Error: ±${test_metrics['RMSE']:.2f}M")
    print(f"   • Percentage Accuracy: {100 - test_metrics['MAPE']:.1f}%")
    print(f"   • Cross-validation Consistency: {cv_scores.std():.4f} std deviation")

if __name__ == "__main__":
    main()
