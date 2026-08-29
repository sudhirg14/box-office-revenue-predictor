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
    "localhost:50051"
)

stub = movie_pb2_grpc.MoviePredictionServiceStub(
    channel
)


# -------------------------------------------------
# STEP 1: Client creates local event
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

    budget_million=100.0,

    release_year=2024,

    runtime_min=140.0,

    critic_rating=8.5,

    audience_rating=8.2,

    review_sentiment=0.75,

    review_volume=25000,

    star_power=0.90,

    social_media_buzz=200000,

    marketing_spend_million=40.0,

    lamport_timestamp=request_time
)


print("\nMovie Information")
print("----------------------------------------")

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

print(
    "\nSending prediction request..."
)

print(
    "Request Lamport Timestamp:",
    request.lamport_timestamp
)


# -------------------------------------------------
# STEP 3: Send request to server
# -------------------------------------------------

try:

    response = stub.PredictRevenue(
        request,
        timeout=10
    )

except grpc.RpcError as e:

    print("\nPrediction server error:")
    print(e)

    channel.close()
    raise


# -------------------------------------------------
# STEP 4: Receive response
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
    response.predicted_revenue,
    "million"
)

print(
    "Message:",
    response.message
)

print(
    "Final Client Lamport Timestamp:",
    clock.get_time()
)


channel.close()