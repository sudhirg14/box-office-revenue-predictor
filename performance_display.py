"""
Simple Performance Metrics Display
A clean, focused display of key model performance metrics
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           explained_variance_score, mean_absolute_percentage_error)

def load_and_evaluate_model():
    """Load model and calculate key performance metrics"""
    print("🎬 MOVIE BOX OFFICE PREDICTOR - PERFORMANCE METRICS")
    print("=" * 60)
    
    try:
        # Load model and data
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        df = pd.read_csv('movie_box_office_uniform_5000.csv')
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Dataset: {df.shape[0]} movies, {df.shape[1]} features")
        
        # Prepare data
        data = df.copy()
        genre_encoded = pd.get_dummies(data['genre'], prefix='genre')
        data = pd.concat([data, genre_encoded], axis=1)
        data = data.drop(columns=['movie_id', 'title', 'genre'])
        
        X = data.drop('box_office_collection', axis=1)
        y = data['box_office_collection']
        
        # Split and predict
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
        explained_var = explained_variance_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'explained_var': explained_var,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def display_metrics(metrics):
    """Display performance metrics in a clean format"""
    if metrics is None:
        return
    
    print("\n📊 PERFORMANCE METRICS")
    print("=" * 60)
    
    # Main accuracy metrics
    print(f"🎯 ACCURACY METRICS:")
    print(f"   R² Score (Coefficient of Determination): {metrics['r2']:.4f}")
    print(f"   Explained Variance Score: {metrics['explained_var']:.4f}")
    print(f"   Percentage Accuracy: {100 - metrics['mape']:.1f}%")
    
    print(f"\n📉 ERROR METRICS:")
    print(f"   Root Mean Square Error (RMSE): ${metrics['rmse']:.2f}M")
    print(f"   Mean Absolute Error (MAE): ${metrics['mae']:.2f}M")
    print(f"   Mean Absolute Percentage Error (MAPE): {metrics['mape']:.1f}%")
    
    print(f"\n🔄 VALIDATION METRICS:")
    print(f"   Cross-Validation R² (5-fold): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"   95% Confidence Interval: [{metrics['cv_mean'] - 1.96*metrics['cv_std']:.4f}, {metrics['cv_mean'] + 1.96*metrics['cv_std']:.4f}]")
    
    # Performance interpretation
    print(f"\n🎯 PERFORMANCE INTERPRETATION:")
    
    # R² interpretation
    if metrics['r2'] >= 0.9:
        r2_level = "Excellent"
        r2_emoji = "🟢"
    elif metrics['r2'] >= 0.8:
        r2_level = "Very Good"
        r2_emoji = "🟡"
    elif metrics['r2'] >= 0.7:
        r2_level = "Good"
        r2_emoji = "🟠"
    elif metrics['r2'] >= 0.6:
        r2_level = "Fair"
        r2_emoji = "🟠"
    else:
        r2_level = "Needs Improvement"
        r2_emoji = "🔴"
    
    print(f"   {r2_emoji} Overall Performance: {r2_level}")
    print(f"   📈 Model explains {metrics['r2']*100:.1f}% of box office variance")
    print(f"   📊 Typical prediction error: ±${metrics['rmse']:.0f}M")
    print(f"   🎯 Average accuracy: {100 - metrics['mape']:.1f}%")
    
    # Business impact
    print(f"\n💼 BUSINESS IMPACT:")
    print(f"   • Investment Decision Support: {'Highly reliable' if metrics['r2'] >= 0.7 else 'Moderately reliable'}")
    print(f"   • Risk Level: {'Low' if metrics['mape'] <= 20 else 'Medium' if metrics['mape'] <= 30 else 'High'}")
    print(f"   • Market Analysis: {'Excellent' if metrics['r2'] >= 0.8 else 'Good'} for trend understanding")

def create_simple_summary(metrics):
    """Create a simple summary file"""
    if metrics is None:
        return
    
    summary = f"""# Model Performance Summary

## Key Metrics
- **R² Score**: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}% variance explained)
- **RMSE**: ${metrics['rmse']:.2f}M (average prediction error)
- **MAE**: ${metrics['mae']:.2f}M (median prediction error)
- **MAPE**: {metrics['mape']:.1f}% (percentage error)
- **Explained Variance**: {metrics['explained_var']:.4f}

## Validation
- **Cross-Validation R²**: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}
- **95% Confidence Interval**: [{metrics['cv_mean'] - 1.96*metrics['cv_std']:.4f}, {metrics['cv_mean'] + 1.96*metrics['cv_std']:.4f}]

## Interpretation
- **Performance Level**: {'Excellent' if metrics['r2'] >= 0.8 else 'Good' if metrics['r2'] >= 0.7 else 'Fair'}
- **Typical Error**: ±${metrics['rmse']:.0f}M for most predictions
- **Accuracy**: {100 - metrics['mape']:.1f}% based on percentage error
- **Business Use**: {'Highly reliable' if metrics['r2'] >= 0.7 else 'Moderately reliable'} for investment decisions

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('SIMPLE_PERFORMANCE_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n📄 Simple summary saved to: SIMPLE_PERFORMANCE_SUMMARY.md")

def main():
    """Main function"""
    metrics = load_and_evaluate_model()
    display_metrics(metrics)
    create_simple_summary(metrics)
    
    print(f"\n✅ Performance metrics display completed!")

if __name__ == "__main__":
    main()



