"""
Movie Box Office Prediction Model Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess_data():
    """Load and preprocess the movie dataset"""
    print("Loading dataset...")
    df = pd.read_csv('data/movie_box_office_uniform_5000.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Create a copy for preprocessing
    data = df.copy()
    
    # Handle categorical variables (genre)
    print("Encoding categorical variables...")
    genre_encoded = pd.get_dummies(data['genre'], prefix='genre')
    data = pd.concat([data, genre_encoded], axis=1)
    
    # Drop original categorical columns and movie_id, title
    columns_to_drop = ['movie_id', 'title', 'genre']
    data = data.drop(columns=columns_to_drop)
    
    print(f"Features after preprocessing: {list(data.columns)}")
    
    return data

def prepare_features_and_target(data):
    """Separate features and target variable"""
    # Target variable
    target = data['box_office_collection']
    
    # Features
    features = data.drop('box_office_collection', axis=1)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    
    return features, target

def train_models(X, y):
    """Train multiple models and select the best one"""
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to try
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        if name == 'Random Forest':
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std()
        }
        
        trained_models[name] = {
            'model': model,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        print(f"{name} Results:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Select best model based on R² score
    best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
    best_model = trained_models[best_model_name]['model']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best R² Score: {results[best_model_name]['R2']:.4f}")
    
    return best_model, scaler, X.columns.tolist(), results

def save_model_and_artifacts(model, scaler, feature_columns, results):
    """Save the trained model and preprocessing artifacts"""
    print("\n" + "="*50)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/best_model.pkl')
    print("✓ Model saved to models/best_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Scaler saved to models/scaler.pkl")
    
    # Save feature columns
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    print("✓ Feature columns saved to models/feature_columns.pkl")
    
    # Save results
    joblib.dump(results, 'models/training_results.pkl')
    print("✓ Training results saved to models/training_results.pkl")
    
    # Save feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('models/feature_importance.csv', index=False)
        print("✓ Feature importance saved to models/feature_importance.csv")
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:25s} {row['importance']:.4f}")

def create_visualizations(results, model, feature_columns):
    """Create visualizations for model analysis"""
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Create visualizations directory
    os.makedirs('static/images', exist_ok=True)
    
    # Model comparison
    plt.figure(figsize=(12, 8))
    
    model_names = list(results.keys())
    r2_scores = [results[name]['R2'] for name in model_names]
    rmse_scores = [results[name]['RMSE'] for name in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² scores
    bars1 = ax1.bar(model_names, r2_scores, color=['skyblue', 'lightcoral'])
    ax1.set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    # RMSE scores
    bars2 = ax2.bar(model_names, rmse_scores, color=['skyblue', 'lightcoral'])
    ax2.set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RMSE (Million $)')
    
    # Add value labels on bars
    for bar, score in zip(bars2, rmse_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Model comparison chart saved")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Plot top 15 features
        top_features = importance_df.tail(15)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('static/images/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature importance chart saved")

def main():
    """Main training pipeline"""
    print("🎬 MOVIE BOX OFFICE PREDICTION MODEL TRAINING")
    print("="*60)
    
    try:
        # Load and preprocess data
        data = load_and_preprocess_data()
        
        # Prepare features and target
        X, y = prepare_features_and_target(data)
        
        # Train models
        best_model, scaler, feature_columns, results = train_models(X, y)
        
        # Save model and artifacts
        save_model_and_artifacts(best_model, scaler, feature_columns, results)
        
        # Create visualizations
        create_visualizations(results, best_model, feature_columns)
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Run 'python app.py' to start the web application")
        print("2. Open http://localhost:5001 in your browser")
        print("3. Start making predictions!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
     main()

