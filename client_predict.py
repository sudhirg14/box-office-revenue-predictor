import grpc
import threading
import time

import movie_pb2
import movie_pb2_grpc

from lamport_clock import LamportClock


# =========================================================
# LOAD BALANCER
# =========================================================

LOAD_BALANCER = "localhost:50050"

# Number of concurrent prediction requests

NUM_REQUESTS = 12


# =========================================================
# SEND ONE PREDICTION REQUEST
# =========================================================

def send_prediction(
    request_id
):

    # Each request has its own Lamport clock

    clock = LamportClock()

    request_time = clock.increment()

    # -----------------------------------------------------
    # Connect to LOAD BALANCER
    # -----------------------------------------------------

    channel = grpc.insecure_channel(
        LOAD_BALANCER
    )

    stub = (
        movie_pb2_grpc
        .MoviePredictionServiceStub(
            channel
        )
    )

    # -----------------------------------------------------
    # Create Movie Request
    # -----------------------------------------------------

    request = movie_pb2.MovieRequest(

        genre="Action",

        budget_million=(
            100.0 + request_id
        ),

        release_year=2024,

        runtime_min=140.0,

        critic_rating=8.5,

        audience_rating=8.2,

        review_sentiment=0.75,

        review_volume=(
            25000 + request_id * 100
        ),

        star_power=0.90,

        social_media_buzz=(
            200000 + request_id * 1000
        ),

        marketing_spend_million=40.0,

        lamport_timestamp=request_time
    )

    print("\n========================================")

    print(
        f"[CLIENT] Request {request_id}"
    )

    print("========================================")

    print(
        "Load Balancer:",
        LOAD_BALANCER
    )

    print(
        "Client Lamport:",
        request_time
    )

    print(
        "Movie Genre:",
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

    # -----------------------------------------------------
    # Send prediction request
    # -----------------------------------------------------

    try:

        print(
            f"[CLIENT] Sending request "
            f"{request_id}..."
        )

        response = stub.PredictRevenue(

            request,

            timeout=20

        )

        # -------------------------------------------------
        # Update Lamport clock
        # -------------------------------------------------

        new_time = clock.update(
            response.lamport_timestamp
        )

        # -------------------------------------------------
        # Display result
        # -------------------------------------------------

        print("\n----------------------------------------")

        print(
            f"[CLIENT] Request "
            f"{request_id} COMPLETED"
        )

        print("----------------------------------------")

        print(
            "Predicted Revenue:",
            response.predicted_revenue,
            "million"
        )

        print(
            "Message:",
            response.message
        )

        print(
            "Response Lamport:",
            response.lamport_timestamp
        )

        print(
            "Updated Client Lamport:",
            new_time
        )

        print("----------------------------------------")

    except grpc.RpcError as e:

        print(
            f"\n[CLIENT] Request "
            f"{request_id} FAILED"
        )

        print(
            "gRPC Error:",
            e
        )

    finally:

        channel.close()


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n========================================")
    print(
        "MOVIE BOX OFFICE PREDICTION CLIENT"
    )
    print(
        "DISTRIBUTED COMPUTING - EXPERIMENT 6"
    )
    print("========================================")

    print(
        "Load Balancer:",
        LOAD_BALANCER
    )

    print(
        "Strategy:",
        "Least Connections"
    )

    print(
        "Number of Requests:",
        NUM_REQUESTS
    )

    print("========================================\n")

    threads = []

    start_time = time.time()

    # -----------------------------------------------------
    # Create concurrent requests
    # -----------------------------------------------------

    for request_id in range(
        1,
        NUM_REQUESTS + 1
    ):

        thread = threading.Thread(

            target=send_prediction,

            args=(request_id,)

        )

        threads.append(
            thread
        )

        thread.start()

        # Small delay between arrivals

        time.sleep(
            0.15
        )

    # -----------------------------------------------------
    # Wait for every request
    # -----------------------------------------------------

    for thread in threads:

        thread.join()

    elapsed_time = (
        time.time() - start_time
    )

    print("\n========================================")
    print(
        "ALL REQUESTS COMPLETED"
    )
    print("========================================")

    print(
        "Total Requests:",
        NUM_REQUESTS
    )

    print(
        f"Total Execution Time: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        "Load Balancing Strategy:",
        "Least Connections"
    )

    print(
        "Experiment 6 Completed Successfully."
    )

    print("========================================")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    main()