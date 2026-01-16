"""
Demo script to test the Movie Box Office Prediction API
"""

import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5001"
    
    print("🎬 MOVIE BOX OFFICE PREDICTOR - API DEMO")
    print("=" * 50)
    
    # Test 1: Get statistics
    print("\n1. Testing /api/stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ Statistics retrieved successfully!")
            print(f"   Total Movies: {stats['total_movies']:,}")
            print(f"   Average Box Office: ${stats['avg_box_office']}M")
            print(f"   Max Box Office: ${stats['max_box_office']}M")
            print(f"   Year Range: {stats['year_range']['min']}-{stats['year_range']['max']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on http://localhost:5000")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Test 2: Get feature importance
    print("\n2. Testing /api/features endpoint...")
    try:
        response = requests.get(f"{base_url}/api/features")
        if response.status_code == 200:
            features = response.json()
            print("✅ Feature importance retrieved successfully!")
            print("   Top 5 Most Important Features:")
            for i, (feature, importance) in enumerate(list(features.items())[:5], 1):
                print(f"   {i}. {feature}: {importance:.4f}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Make a prediction
    print("\n3. Testing /predict endpoint...")
    
    # Sample movie data
    sample_movies = [
        {
            "name": "High-Budget Action Blockbuster",
            "data": {
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
        },
        {
            "name": "Low-Budget Indie Drama",
            "data": {
                "genre": "Drama",
                "budget_million": 5.0,
                "release_year": 2024,
                "runtime_min": 95,
                "critic_rating": 8.5,
                "audience_rating": 7.0,
                "review_sentiment": 0.3,
                "review_volume": 5000,
                "star_power": 0.2,
                "social_media_buzz": 10000,
                "marketing_spend_million": 2.0
            }
        },
        {
            "name": "Horror Movie",
            "data": {
                "genre": "Horror",
                "budget_million": 20.0,
                "release_year": 2024,
                "runtime_min": 90,
                "critic_rating": 6.0,
                "audience_rating": 6.5,
                "review_sentiment": -0.2,
                "review_volume": 8000,
                "star_power": 0.4,
                "social_media_buzz": 30000,
                "marketing_spend_million": 8.0
            }
        }
    ]
    
    for movie in sample_movies:
        print(f"\n   Predicting for: {movie['name']}")
        try:
            response = requests.post(f"{base_url}/predict", json=movie['data'])
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Predicted Box Office: ${result['prediction']:.2f}M")
                print(f"   ✅ Confidence: {result['confidence']:.1f}%")
            else:
                print(f"   ❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API DEMO COMPLETED!")
    print("=" * 50)
    print("\nTo use the web interface:")
    print("1. Open your browser")
    print("2. Go to: http://localhost:5000")
    print("3. Fill in the prediction form")
    print("4. Click 'Predict Box Office Collection'")

if __name__ == "__main__":
    test_api()
