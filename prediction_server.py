from concurrent import futures
from lamport_clock import LamportClock

import os

import grpc
import joblib
import numpy as np
import pandas as pd



import movie_pb2
import movie_pb2_grpc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


class MoviePredictionService(
    movie_pb2_grpc.MoviePredictionServiceServicer
):

    def __init__(self):

        # -------------------------------------------------
        # Lamport Logical Clock
        # -------------------------------------------------
        self.clock = LamportClock()

        # -------------------------------------------------
        # Load trained ML model
        # -------------------------------------------------
        print("Loading trained prediction model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not os.path.exists(FEATURE_COLUMNS_PATH):
            raise FileNotFoundError(
                f"Feature columns not found: "
                f"{FEATURE_COLUMNS_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)
        self.feature_columns = joblib.load(
            FEATURE_COLUMNS_PATH
        )

        # Scaler is needed only if the selected model
        # was trained using scaled features.
        self.scaler = None

        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)

        print("✓ ML model loaded successfully")
        print(
            f"✓ Number of model features: "
            f"{len(self.feature_columns)}"
        )

    # =====================================================
    # FEATURE PREPARATION
    # =====================================================

    def prepare_features(self, request):

        # Create DataFrame with exactly the fields used
        # by the training pipeline.

        input_data = {
            "budget_million": request.budget_million,
            "release_year": request.release_year,
            "runtime_min": request.runtime_min,
            "critic_rating": request.critic_rating,
            "audience_rating": request.audience_rating,
            "review_sentiment": request.review_sentiment,
            "review_volume": request.review_volume,
            "star_power": request.star_power,
            "social_media_buzz": request.social_media_buzz,
            "marketing_spend_million":
                request.marketing_spend_million
        }

        df = pd.DataFrame([input_data])

        # -------------------------------------------------
        # Genre one-hot encoding
        # -------------------------------------------------

        genre = request.genre

        for feature in self.feature_columns:

            if feature.startswith("genre_"):
                df[feature] = (
                    1 if feature == f"genre_{genre}"
                    else 0
                )

        # -------------------------------------------------
        # Make sure every training feature exists
        # and is in exactly the same order.
        # -------------------------------------------------

        for feature in self.feature_columns:

            if feature not in df.columns:
                df[feature] = 0

        df = df[self.feature_columns]

        return df

    # =====================================================
    # gRPC PREDICTION
    # =====================================================

    def PredictRevenue(self, request, context):

        # -------------------------------------------------
        # STEP 1: Receive request
        # -------------------------------------------------

        current_time = self.clock.update(
            request.lamport_timestamp
        )

        print("\n----------------------------------------")
        print("Movie Prediction Request Received")
        print("----------------------------------------")

        print(
            "Received Lamport Timestamp:",
            request.lamport_timestamp
        )

        print(
            "Server Lamport Timestamp:",
            current_time
        )

        print("\nMovie Features:")
        print("Genre:", request.genre)
        print("Budget:", request.budget_million)
        print("Release Year:", request.release_year)
        print("Runtime:", request.runtime_min)
        print("Critic Rating:", request.critic_rating)
        print("Audience Rating:", request.audience_rating)
        print("Review Sentiment:", request.review_sentiment)
        print("Review Volume:", request.review_volume)
        print("Star Power:", request.star_power)
        print("Social Media Buzz:", request.social_media_buzz)
        print(
            "Marketing Spend:",
            request.marketing_spend_million
        )

        # -------------------------------------------------
        # STEP 2: Prepare features
        # -------------------------------------------------

        try:

            features = self.prepare_features(request)

            print("\nPrepared Features:")
            print(features)

            # -------------------------------------------------
            # IMPORTANT:
            # Random Forest in train_model.py is trained
            # using UN-SCALED features.
            #
            # XGBoost is trained using SCALED features.
            #
            # We therefore determine which model was saved
            # from its class.
            # -------------------------------------------------

            model_name = type(self.model).__name__

            if model_name == "XGBRegressor":

                if self.scaler is None:
                    raise RuntimeError(
                        "Scaler is required for XGBoost "
                        "but scaler.pkl was not found."
                    )

                model_features = self.scaler.transform(
                    features
                )

            else:

                # Random Forest:
                # DO NOT SCALE.
                model_features = features

            # -------------------------------------------------
            # STEP 3: Actual ML prediction
            # -------------------------------------------------

            predicted_revenue = float(
                self.model.predict(model_features)[0]
            )

            # Prevent negative revenue caused by model
            # extrapolation.
            predicted_revenue = max(
                0.0,
                predicted_revenue
            )

            print(
                "\nActual ML Prediction:",
                predicted_revenue
            )

        except Exception as e:

            print(
                "Prediction error:",
                str(e)
            )

            context.set_code(
                grpc.StatusCode.INTERNAL
            )

            context.set_details(
                f"Prediction failed: {str(e)}"
            )

            return movie_pb2.PredictionResponse(
                predicted_revenue=0.0,
                message=f"Prediction failed: {str(e)}",
                lamport_timestamp=self.clock.increment()
            )

        # -------------------------------------------------
        # STEP 4: Increment Lamport clock
        # -------------------------------------------------

        response_time = self.clock.increment()

        print(
            "Response Lamport Timestamp:",
            response_time
        )

        # -------------------------------------------------
        # STEP 5: Return actual prediction
        # -------------------------------------------------

        return movie_pb2.PredictionResponse(
            predicted_revenue=predicted_revenue,
            message="Prediction completed successfully!",
            lamport_timestamp=response_time
        )


# =========================================================
# SERVER
# =========================================================

def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10
        )
    )

    movie_pb2_grpc.add_MoviePredictionServiceServicer_to_server(
        MoviePredictionService(),
        server
    )

    server.add_insecure_port(
        "[::]:50051"
    )

    server.start()

    print(
        "\n========================================"
    )
    print(
        "Movie Prediction gRPC Server"
    )
    print(
        "========================================"
    )
    print(
        "Prediction Server running on port 50051"
    )
    print(
        "Actual ML Model: ENABLED"
    )
    print(
        "Lamport Clock: ENABLED"
    )
    print(
        "========================================\n"
    )

    server.wait_for_termination()


if __name__ == "__main__":
    serve()