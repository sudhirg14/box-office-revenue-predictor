import grpc

import movie_pb2
import movie_pb2_grpc

from lamport_clock import LamportClock


# -------------------------------------------------
# Create Lamport Clock for Client
# -------------------------------------------------

clock = LamportClock()


# -------------------------------------------------
# Connect to Prediction Server
# -------------------------------------------------

channel = grpc.insecure_channel(
    'localhost:50051'
)

stub = movie_pb2_grpc.MoviePredictionServiceStub(
    channel
)


# -------------------------------------------------
# STEP 1: Client creates a local event
# -------------------------------------------------

request_time = clock.increment()

print("\n========================================")
print("MOVIE BOX OFFICE PREDICTION CLIENT")
print("========================================")

print(
    "Client Lamport Timestamp:",
    request_time
)


# -------------------------------------------------
# STEP 2: Create request
# -------------------------------------------------

request = movie_pb2.MovieRequest(

    genre="Action",

    budget=100000000,

    runtime=140,

    imdb_rating=8.5,

    lamport_timestamp=request_time
)


print("\nSending prediction request...")
print(
    "Request Lamport Timestamp:",
    request.lamport_timestamp
)


# -------------------------------------------------
# STEP 3: Send request to server
# -------------------------------------------------

response = stub.PredictRevenue(request)


# -------------------------------------------------
# STEP 4: Receive response and update clock
# -------------------------------------------------

new_time = clock.update(
    response.lamport_timestamp
)


print("\nResponse received from server.")

print(
    "Server Lamport Timestamp:",
    response.lamport_timestamp
)

print(
    "Updated Client Lamport Timestamp:",
    new_time
)


# -------------------------------------------------
# STEP 5: Display prediction
# -------------------------------------------------

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print(
    "Predicted Revenue:",
    response.predicted_revenue
)

print(
    "Message:",
    response.message
)

print(
    "Final Client Lamport Timestamp:",
    clock.get_time()
)