"""
Movie Box Office Prediction Model Performance Analysis
This script displays comprehensive performance metrics of the trained model.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score, max_error
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """Load the trained model, scaler, and data"""
    print("🔍 LOADING MODEL AND DATA")
    print("=" * 50)
    
    try:
        # Load model and preprocessing objects
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        training_results = joblib.load('models/training_results.pkl')
        
        # Load dataset
        df = pd.read_csv('data/movie_box_office_uniform_5000.csv')
        
        print("✅ Model and data loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🧠 Model type: {type(model).__name__}")
        
        return model, scaler, feature_columns, training_results, df
        
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        print("Make sure to run train_model.py first!")
        return None, None, None, None, None

def prepare_data(df):
    """Prepare data for model evaluation"""
    print("\n🛠️ PREPARING DATA FOR EVALUATION")
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
    print("\n📈 CALCULATING PERFORMANCE METRICS")
    print("=" * 50)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics for training set
    train_metrics = {
        'R² Score': r2_score(y_train, y_pred_train),
        'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'MAE': mean_absolute_error(y_train, y_pred_train),
        'Explained Variance': explained_variance_score(y_train, y_pred_train),
        'Max Error': max_error(y_train, y_pred_train)
    }
    
    # Calculate metrics for test set
    test_metrics = {
        'R² Score': r2_score(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'Explained Variance': explained_variance_score(y_test, y_pred_test),
        'Max Error': max_error(y_test, y_pred_test)
    }
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate residuals
    residuals = y_test - y_pred_test
    
    print("✅ Performance metrics calculated!")
    
    return train_metrics, test_metrics, cv_scores, residuals, y_test, y_pred_test

def display_performance_summary(train_metrics, test_metrics, cv_scores):
    """Display comprehensive performance summary"""
    print("\n🎯 MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame({
        'Training Set': [f"{train_metrics['R² Score']:.4f}",
                        f"${train_metrics['RMSE']:.2f}M",
                        f"${train_metrics['MAE']:.2f}M",
                        f"{train_metrics['Explained Variance']:.4f}",
                        f"${train_metrics['Max Error']:.2f}M"],
        'Test Set': [f"{test_metrics['R² Score']:.4f}",
                    f"${test_metrics['RMSE']:.2f}M",
                    f"${test_metrics['MAE']:.2f}M",
                    f"{test_metrics['Explained Variance']:.4f}",
                    f"${test_metrics['Max Error']:.2f}M"]
    }, index=['R² Score', 'RMSE', 'MAE', 'Explained Variance', 'Max Error'])
    
    print(metrics_df.to_string())
    
    # Cross-validation results
    print(f"\n🔄 CROSS-VALIDATION RESULTS (5-fold)")
    print(f"   Mean R² Score: {cv_scores.mean():.4f}")
    print(f"   Std Deviation: {cv_scores.std():.4f}")
    print(f"   95% CI: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]")
    
    # Performance interpretation
    print(f"\n📊 PERFORMANCE INTERPRETATION")
    print("=" * 60)
    
    r2_test = test_metrics['R² Score']
    rmse_test = test_metrics['RMSE']
    
    if r2_test >= 0.8:
        performance_level = "Excellent"
        color = "🟢"
    elif r2_test >= 0.7:
        performance_level = "Very Good"
        color = "🟡"
    elif r2_test >= 0.6:
        performance_level = "Good"
        color = "🟠"
    else:
        performance_level = "Needs Improvement"
        color = "🔴"
    
    print(f"{color} Overall Performance: {performance_level}")
    print(f"📈 Model explains {r2_test*100:.1f}% of the variance in box office collections")
    print(f"📉 Average prediction error: ${rmse_test:.2f}M")
    print(f"🎯 Typical prediction accuracy: ±${rmse_test:.0f}M")

def display_feature_importance(model, feature_columns):
    """Display feature importance analysis"""
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            percentage = row['Importance'] * 100
            bar = "█" * int(percentage * 2)  # Visual bar
            print(f"{i:2d}. {row['Feature']:25s} {row['Importance']:.4f} {percentage:5.1f}% {bar}")
        
        # Feature categories analysis
        print(f"\n📊 FEATURE CATEGORY ANALYSIS")
        print("-" * 40)
        
        # Categorize features
        budget_features = [f for f in feature_columns if 'budget' in f.lower()]
        rating_features = [f for f in feature_columns if 'rating' in f.lower()]
        marketing_features = [f for f in feature_columns if any(x in f.lower() for x in ['marketing', 'social', 'review'])]
        genre_features = [f for f in feature_columns if f.startswith('genre_')]
        other_features = [f for f in feature_columns if not any(x in f.lower() for x in ['budget', 'rating', 'marketing', 'social', 'review']) and not f.startswith('genre_')]
        
        categories = {
            'Financial': budget_features + [f for f in feature_columns if 'marketing_spend' in f],
            'Quality Metrics': rating_features + [f for f in feature_columns if 'sentiment' in f.lower()],
            'Marketing & Buzz': marketing_features,
            'Genre': genre_features,
            'Other': other_features
        }
        
        for category, features in categories.items():
            if features:
                category_importance = sum(importance_df[importance_df['Feature'].isin(features)]['Importance'])
                percentage = category_importance * 100
                print(f"{category:20s}: {category_importance:.4f} ({percentage:5.1f}%)")
    else:
        print("❌ Feature importance not available for this model type")

def create_performance_visualizations(train_metrics, test_metrics, residuals, y_test, y_pred_test, feature_columns, model):
    """Create comprehensive performance visualizations"""
    print("\n📊 CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 50)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Performance Metrics Comparison
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['R² Score', 'RMSE', 'MAE', 'Explained Variance']
    train_values = [train_metrics[m] for m in metrics]
    test_values = [test_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance: Training vs Test')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Residuals Plot
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(y_pred_test, residuals, alpha=0.6, color='purple')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(y_test, y_pred_test, alpha=0.6, color='green')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual Box Office (Million $)')
    ax3.set_ylabel('Predicted Box Office (Million $)')
    ax3.set_title('Actual vs Predicted Values')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals Distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--')
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Q Plot for residuals
    ax5 = plt.subplot(3, 3, 5)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot of Residuals')
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature Importance
    if hasattr(model, 'feature_importances_'):
        ax6 = plt.subplot(3, 3, 6)
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        top_features = importance_df.tail(10)
        ax6.barh(range(len(top_features)), top_features['Importance'], color='lightblue')
        ax6.set_yticks(range(len(top_features)))
        ax6.set_yticklabels(top_features['Feature'])
        ax6.set_xlabel('Importance')
        ax6.set_title('Top 10 Feature Importance')
        ax6.grid(True, alpha=0.3)
    
    # 7. Error Analysis
    ax7 = plt.subplot(3, 3, 7)
    errors = np.abs(residuals)
    error_bins = [0, 25, 50, 75, 100, 150, 200, float('inf')]
    error_labels = ['<25M', '25-50M', '50-75M', '75-100M', '100-150M', '150-200M', '>200M']
    error_counts = pd.cut(errors, bins=error_bins, labels=error_labels, include_lowest=True).value_counts()
    
    wedges, texts, autotexts = ax7.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%', startangle=90)
    ax7.set_title('Distribution of Prediction Errors')
    
    # 8. Performance by Box Office Range
    ax8 = plt.subplot(3, 3, 8)
    box_office_ranges = ['<100M', '100-200M', '200-300M', '300-400M', '400-500M', '>500M']
    range_bins = [0, 100, 200, 300, 400, 500, float('inf')]
    y_test_ranges = pd.cut(y_test, bins=range_bins, labels=box_office_ranges, include_lowest=True)
    
    range_mae = []
    for range_label in box_office_ranges:
        mask = y_test_ranges == range_label
        if mask.sum() > 0:
            range_mae.append(mean_absolute_error(y_test[mask], y_pred_test[mask]))
        else:
            range_mae.append(0)
    
    bars = ax8.bar(range(len(box_office_ranges)), range_mae, color='lightgreen', alpha=0.7)
    ax8.set_xlabel('Box Office Range')
    ax8.set_ylabel('Mean Absolute Error')
    ax8.set_title('Prediction Error by Box Office Range')
    ax8.set_xticks(range(len(box_office_ranges)))
    ax8.set_xticklabels(box_office_ranges, rotation=45)
    ax8.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars, range_mae):
        if mae > 0:
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{mae:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 9. Model Confidence Analysis
    ax9 = plt.subplot(3, 3, 9)
    # Calculate prediction confidence based on distance from mean
    mean_box_office = np.mean(y_test)
    confidence = 100 - np.abs(y_pred_test - mean_box_office) / mean_box_office * 100
    confidence = np.clip(confidence, 0, 100)
    
    ax9.hist(confidence, bins=20, alpha=0.7, color='gold', edgecolor='black')
    ax9.set_xlabel('Prediction Confidence (%)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Distribution of Prediction Confidence')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance visualizations created and saved!")
    print("📁 Saved to: static/images/model_performance_analysis.png")

def generate_performance_report(train_metrics, test_metrics, cv_scores, feature_columns, model):
    """Generate a comprehensive performance report"""
    print("\n📋 GENERATING PERFORMANCE REPORT")
    print("=" * 50)
    
    report = f"""
# 🎬 Movie Box Office Predictor - Model Performance Report

## 📊 Executive Summary
- **Model Type**: {type(model).__name__}
- **Overall Performance**: {'Excellent' if test_metrics['R² Score'] >= 0.8 else 'Very Good' if test_metrics['R² Score'] >= 0.7 else 'Good'}
- **Prediction Accuracy**: {test_metrics['R² Score']*100:.1f}% variance explained
- **Average Error**: ±${test_metrics['RMSE']:.2f}M

## 🎯 Key Performance Metrics

### Training Set Performance
- **R² Score**: {train_metrics['R² Score']:.4f}
- **RMSE**: ${train_metrics['RMSE']:.2f}M
- **MAE**: ${train_metrics['MAE']:.2f}M
- **Explained Variance**: {train_metrics['Explained Variance']:.4f}

### Test Set Performance
- **R² Score**: {test_metrics['R² Score']:.4f}
- **RMSE**: ${test_metrics['RMSE']:.2f}M
- **MAE**: ${test_metrics['MAE']:.2f}M
- **Explained Variance**: {test_metrics['Explained Variance']:.4f}

### Cross-Validation Results (5-fold)
- **Mean R² Score**: {cv_scores.mean():.4f}
- **Standard Deviation**: {cv_scores.std():.4f}
- **95% Confidence Interval**: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, {cv_scores.mean() + 1.96*cv_scores.std():.4f}]

## 🔍 Feature Importance Analysis
"""
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        report += "\n### Top 10 Most Important Features\n"
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            report += f"{i:2d}. **{row['Feature']}**: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)\n"
    
    report += f"""

## 📈 Model Interpretation

### Strengths
- {'High' if test_metrics['R² Score'] >= 0.7 else 'Moderate'} predictive accuracy
- {'Consistent' if cv_scores.std() < 0.05 else 'Acceptable'} cross-validation performance
- {'Strong' if test_metrics['Explained Variance'] >= 0.7 else 'Good'} variance explanation

### Areas for Improvement
- {'None identified - excellent performance!' if test_metrics['R² Score'] >= 0.8 else 'Consider feature engineering or model tuning' if test_metrics['R² Score'] >= 0.6 else 'Significant improvement needed'}
- {'Stable' if abs(train_metrics['R² Score'] - test_metrics['R² Score']) < 0.1 else 'Some overfitting detected'}

## 🎯 Business Impact
- **Prediction Reliability**: {test_metrics['R² Score']*100:.1f}% of box office variance explained
- **Typical Error Range**: ±${test_metrics['RMSE']:.0f}M for most predictions
- **Investment Decision Support**: {'Highly reliable' if test_metrics['R² Score'] >= 0.7 else 'Moderately reliable'} for budget allocation decisions

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Save report to file
    with open('MODEL_PERFORMANCE_REPORT.md', 'w') as f:
        f.write(report)
    
    print("✅ Performance report generated!")
    print("📁 Saved to: MODEL_PERFORMANCE_REPORT.md")
    
    return report

def main():
    """Main function to run the complete performance analysis"""
    print("🎬 MOVIE BOX OFFICE PREDICTOR - MODEL PERFORMANCE ANALYSIS")
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
    
    # Display performance summary
    display_performance_summary(train_metrics, test_metrics, cv_scores)
    
    # Display feature importance
    display_feature_importance(model, feature_columns)
    
    # Create visualizations
    create_performance_visualizations(
        train_metrics, test_metrics, residuals, y_test, y_pred_test, feature_columns, model
    )
    
    # Generate report
    report = generate_performance_report(train_metrics, test_metrics, cv_scores, feature_columns, model)
    
    print("\n" + "=" * 70)
    print("✅ MODEL PERFORMANCE ANALYSIS COMPLETED!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print("   • static/images/model_performance_analysis.png - Performance visualizations")
    print("   • MODEL_PERFORMANCE_REPORT.md - Comprehensive performance report")
    print("\n🎯 Key Insights:")
    print(f"   • Model explains {test_metrics['R² Score']*100:.1f}% of box office variance")
    print(f"   • Average prediction error: ±${test_metrics['RMSE']:.2f}M")
    print(f"   • Cross-validation consistency: {cv_scores.std():.4f} std deviation")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        top_feature = importance_df.iloc[0]
        print(f"   • Most important feature: {top_feature['Feature']} ({top_feature['Importance']*100:.1f}%)")

if __name__ == "__main__":
    main()
