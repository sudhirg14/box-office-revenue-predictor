from concurrent import futures
from lamport_clock import LamportClock

import os
import sys
import time
import random

import grpc
import joblib
import pandas as pd

import movie_pb2
import movie_pb2_grpc


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.pkl"
)


class MoviePredictionService(
    movie_pb2_grpc.MoviePredictionServiceServicer
):

    def __init__(self, worker_name):

        # -------------------------------------------------
        # Worker Information
        # -------------------------------------------------

        self.worker_name = worker_name

        # -------------------------------------------------
        # Lamport Logical Clock
        # -------------------------------------------------

        self.clock = LamportClock()

        # -------------------------------------------------
        # Load trained ML model
        # -------------------------------------------------

        print(
            f"[{self.worker_name}] "
            "Loading trained prediction model..."
        )

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not os.path.exists(FEATURE_COLUMNS_PATH):

            raise FileNotFoundError(
                f"Feature columns not found: "
                f"{FEATURE_COLUMNS_PATH}"
            )

        self.model = joblib.load(
            MODEL_PATH
        )

        self.feature_columns = joblib.load(
            FEATURE_COLUMNS_PATH
        )

        # Scaler is needed only if the selected
        # model was trained using scaled features.

        self.scaler = None

        if os.path.exists(SCALER_PATH):

            self.scaler = joblib.load(
                SCALER_PATH
            )

        print(
            f"[{self.worker_name}] "
            "✓ ML model loaded successfully"
        )

        print(
            f"[{self.worker_name}] "
            f"✓ Number of model features: "
            f"{len(self.feature_columns)}"
        )

    # =====================================================
    # FEATURE PREPARATION
    # =====================================================

    def prepare_features(self, request):

        # Create DataFrame with exactly the fields
        # used by the training pipeline.

        input_data = {

            "budget_million":
                request.budget_million,

            "release_year":
                request.release_year,

            "runtime_min":
                request.runtime_min,

            "critic_rating":
                request.critic_rating,

            "audience_rating":
                request.audience_rating,

            "review_sentiment":
                request.review_sentiment,

            "review_volume":
                request.review_volume,

            "star_power":
                request.star_power,

            "social_media_buzz":
                request.social_media_buzz,

            "marketing_spend_million":
                request.marketing_spend_million
        }

        df = pd.DataFrame(
            [input_data]
        )

        # -------------------------------------------------
        # Genre one-hot encoding
        # -------------------------------------------------

        genre = request.genre

        for feature in self.feature_columns:

            if feature.startswith("genre_"):

                df[feature] = (

                    1
                    if feature == f"genre_{genre}"
                    else 0

                )

        # -------------------------------------------------
        # Make sure every training feature exists
        # -------------------------------------------------

        for feature in self.feature_columns:

            if feature not in df.columns:

                df[feature] = 0

        # Ensure exact feature order

        df = df[
            self.feature_columns
        ]

        return df

    # =====================================================
    # gRPC PREDICTION
    # =====================================================

    def PredictRevenue(
        self,
        request,
        context
    ):

        # -------------------------------------------------
        # STEP 1: Receive request
        # -------------------------------------------------

        current_time = self.clock.update(
            request.lamport_timestamp
        )

        print("\n========================================")
        print(
            f"[{self.worker_name}] "
            "MOVIE PREDICTION REQUEST"
        )
        print("========================================")

        print(
            "Received Lamport Timestamp:",
            request.lamport_timestamp
        )

        print(
            "Server Lamport Timestamp:",
            current_time
        )

        print(
            "Worker:",
            self.worker_name
        )

        # -------------------------------------------------
        # Experiment 6:
        # Simulate variable processing time
        # -------------------------------------------------

        processing_time = random.uniform(
            1.0,
            3.0
        )

        print(
            f"[{self.worker_name}] "
            f"Simulated processing time: "
            f"{processing_time:.2f} seconds"
        )

        time.sleep(
            processing_time
        )

        # -------------------------------------------------
        # Display Movie Features
        # -------------------------------------------------

        print("\nMovie Features:")

        print(
            "Genre:",
            request.genre
        )

        print(
            "Budget:",
            request.budget_million
        )

        print(
            "Release Year:",
            request.release_year
        )

        print(
            "Runtime:",
            request.runtime_min
        )

        print(
            "Critic Rating:",
            request.critic_rating
        )

        print(
            "Audience Rating:",
            request.audience_rating
        )

        print(
            "Review Sentiment:",
            request.review_sentiment
        )

        print(
            "Review Volume:",
            request.review_volume
        )

        print(
            "Star Power:",
            request.star_power
        )

        print(
            "Social Media Buzz:",
            request.social_media_buzz
        )

        print(
            "Marketing Spend:",
            request.marketing_spend_million
        )

        # -------------------------------------------------
        # STEP 2: Prepare features
        # -------------------------------------------------

        try:

            features = self.prepare_features(
                request
            )

            print("\nPrepared Features:")
            print(features)

            # -------------------------------------------------
            # Determine model type
            # -------------------------------------------------

            model_name = type(
                self.model
            ).__name__

            if model_name == "XGBRegressor":

                if self.scaler is None:

                    raise RuntimeError(
                        "Scaler is required for XGBoost "
                        "but scaler.pkl was not found."
                    )

                model_features = (
                    self.scaler.transform(
                        features
                    )
                )

            else:

                # Random Forest:
                # DO NOT SCALE.

                model_features = features

            # -------------------------------------------------
            # STEP 3: Actual ML prediction
            # -------------------------------------------------

            predicted_revenue = float(

                self.model.predict(
                    model_features
                )[0]

            )

            # Prevent negative revenue

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
                f"[{self.worker_name}] "
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

                message=(
                    f"Prediction failed: "
                    f"{str(e)}"
                ),

                lamport_timestamp=
                    self.clock.increment()
            )

        # -------------------------------------------------
        # STEP 4: Increment Lamport clock
        # -------------------------------------------------

        response_time = self.clock.increment()

        print(
            f"[{self.worker_name}] "
            "Response Lamport Timestamp:",
            response_time
        )

        # -------------------------------------------------
        # STEP 5: Return actual prediction
        # -------------------------------------------------

        print(
            f"[{self.worker_name}] "
            "Request completed successfully"
        )

        return movie_pb2.PredictionResponse(

            predicted_revenue=
                predicted_revenue,

            message=(
                f"Prediction completed successfully "
                f"by {self.worker_name}!"
            ),

            lamport_timestamp=
                response_time
        )


# =========================================================
# SERVER
# =========================================================

def serve(port):

    worker_name = (
        f"Prediction-Server-{port}"
    )

    server = grpc.server(

        futures.ThreadPoolExecutor(
            max_workers=10
        )

    )

    movie_pb2_grpc.add_MoviePredictionServiceServicer_to_server(

        MoviePredictionService(
            worker_name
        ),

        server

    )

    server.add_insecure_port(
        f"[::]:{port}"
    )

    server.start()

    print("\n========================================")
    print(
        "MOVIE PREDICTION gRPC SERVER"
    )
    print("========================================")

    print(
        "Worker:",
        worker_name
    )

    print(
        "Port:",
        port
    )

    print(
        "Actual ML Model: ENABLED"
    )

    print(
        "Lamport Clock: ENABLED"
    )

    print(
        "Least Connections Backend: ENABLED"
    )

    print("========================================\n")

    try:

        server.wait_for_termination()

    except KeyboardInterrupt:

        print(
            f"\n[{worker_name}] "
            "Server shutting down..."
        )

        server.stop(0)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python prediction_server.py PORT"
        )

        print(
            "\nExample:"
        )

        print(
            "python prediction_server.py 50051"
        )

        sys.exit(1)

    port = int(
        sys.argv[1]
    )

    serve(port)