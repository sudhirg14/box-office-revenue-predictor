import hashlib
import json
import sys

import grpc
import redis

import movie_pb2
import movie_pb2_grpc


# =========================================================
# CONFIGURATION
# =========================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379

BACKEND_SERVICE_ADDR = "localhost:50051"

# Short TTL so cache behavior is easy to demonstrate
CACHE_TTL_SECONDS = 10

FRESH_KEY_PREFIX = "fresh:movie:"
STALE_KEY_PREFIX = "stale:movie:"


# =========================================================
# REDIS CONNECTION
# =========================================================

def get_redis():
    """
    Create a fresh Redis connection for each request.

    Short timeouts allow the gateway to detect a Redis
    failure quickly instead of hanging.
    """

    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        socket_connect_timeout=1,
        socket_timeout=1
    )


# =========================================================
# CREATE MOVIE REQUEST
# =========================================================

def create_movie_request(request_id):
    """
    Creates the same type of movie prediction request
    used by the existing project client.
    """

    return movie_pb2.MovieRequest(
        genre="Action",

        budget_million=100.0 + request_id,

        release_year=2024,

        runtime_min=140.0,

        critic_rating=8.5,

        audience_rating=8.2,

        review_sentiment=0.75,

        review_volume=25000 + request_id * 100,

        star_power=0.90,

        social_media_buzz=200000 + request_id * 1000,

        marketing_spend_million=40.0,

        # Lamport timestamp is NOT part of the cache key.
        # It changes between requests even when the movie
        # prediction input is the same.
        lamport_timestamp=1
    )


# =========================================================
# CREATE CACHE KEY
# =========================================================

def create_cache_key(request):
    """
    Generate a deterministic cache key from the actual
    movie prediction features.

    Lamport timestamp is deliberately excluded because
    it represents distributed-system timing rather than
    movie prediction data.
    """

    cache_data = {
        "genre": request.genre,
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

    serialized = json.dumps(
        cache_data,
        sort_keys=True
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# =========================================================
# BACKEND REQUEST
# =========================================================

def fetch_from_backend(request):
    """
    Send the actual prediction request to the existing
    gRPC movie prediction backend.
    """

    with grpc.insecure_channel(
        BACKEND_SERVICE_ADDR
    ) as channel:

        stub = movie_pb2_grpc.MoviePredictionServiceStub(
            channel
        )

        response = stub.PredictRevenue(
            request,
            timeout=5
        )

        return {
            "predicted_revenue":
                response.predicted_revenue,

            "message":
                response.message,

            "lamport_timestamp":
                response.lamport_timestamp
        }


# =========================================================
# CACHE-ASIDE LOGIC
# =========================================================

def get_prediction(request):

    cache_id = create_cache_key(request)

    fresh_key = (
        FRESH_KEY_PREFIX + cache_id
    )

    stale_key = (
        STALE_KEY_PREFIX + cache_id
    )

    redis_client = None

    # -----------------------------------------------------
    # STEP 1: TRY REDIS
    # -----------------------------------------------------

    try:

        redis_client = get_redis()

        cached_data = redis_client.get(
            fresh_key
        )

        if cached_data:

            print(
                f"[Gateway] CACHE HIT"
            )

            return (
                json.loads(cached_data),
                "cache-hit"
            )

        print(
            "[Gateway] CACHE MISS "
            "-- querying backend"
        )

    except redis.exceptions.RedisError as e:

        print(
            f"[Gateway] REDIS UNAVAILABLE "
            f"({e})"
        )

        print(
            "[Gateway] Falling back "
            "directly to backend"
        )

        redis_client = None

    # -----------------------------------------------------
    # STEP 2: BACKEND REQUEST
    # -----------------------------------------------------

    try:

        data = fetch_from_backend(request)

        print(
            "[Gateway] Prediction fetched "
            "from BACKEND"
        )

        # -------------------------------------------------
        # STEP 3: SAVE RESULT TO REDIS
        # -------------------------------------------------

        if redis_client is not None:

            try:

                # Normal cache with TTL
                redis_client.set(
                    fresh_key,
                    json.dumps(data),
                    ex=CACHE_TTL_SECONDS
                )

                # Permanent stale backup
                redis_client.set(
                    stale_key,
                    json.dumps(data)
                )

                print(
                    "[Gateway] Prediction stored "
                    "in Redis"
                )

            except redis.exceptions.RedisError:

                print(
                    "[Gateway] Redis write failed"
                )

                print(
                    "[Gateway] Continuing "
                    "without caching"
                )

        return data, "backend"

    # -----------------------------------------------------
    # STEP 4: BACKEND FAILED
    # -----------------------------------------------------

    except grpc.RpcError as e:

        print(
            f"[Gateway] BACKEND UNAVAILABLE "
            f"({e.code()})"
        )

        print(
            "[Gateway] Checking stale cache..."
        )

        # -------------------------------------------------
        # STEP 5: STALE CACHE FALLBACK
        # -------------------------------------------------

        if redis_client is not None:

            try:

                stale_data = redis_client.get(
                    stale_key
                )

                if stale_data:

                    print(
                        "[Gateway] Serving "
                        "STALE cached prediction"
                    )

                    return (
                        json.loads(stale_data),
                        "stale-fallback"
                    )

            except redis.exceptions.RedisError:

                pass

        # -------------------------------------------------
        # STEP 6: EVERYTHING FAILED
        # -------------------------------------------------

        raise RuntimeError(
            "Prediction unavailable: "
            "both backend and cache failed"
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    request_id = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else 1
    )

    print("\n========================================")
    print("EXPERIMENT 8")
    print("REDIS CACHING FOR FAULT TOLERANCE")
    print("========================================")

    print(
        f"Movie Request ID: {request_id}"
    )

    request = create_movie_request(
        request_id
    )

    print(
        f"Genre: {request.genre}"
    )

    print(
        f"Budget: "
        f"{request.budget_million} million"
    )

    print(
        f"Release Year: "
        f"{request.release_year}"
    )

    try:

        data, source = get_prediction(
            request
        )

        print("\n----------------------------------------")

        print(
            f"Result Source: {source}"
        )

        print(
            "Predicted Revenue:",
            data["predicted_revenue"],
            "million"
        )

        print(
            "Message:",
            data["message"]
        )

        print(
            "Lamport Timestamp:",
            data["lamport_timestamp"]
        )

        print("----------------------------------------")

    except Exception as e:

        print(
            f"\n[Gateway] ERROR: {e}"
        )