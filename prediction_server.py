from concurrent import futures
from lamport_clock import LamportClock
import grpc
import movie_pb2
import movie_pb2_grpc


class MoviePredictionService(
    movie_pb2_grpc.MoviePredictionServiceServicer
):

    def __init__(self):
        # Create Lamport Logical Clock for the server
        self.clock = LamportClock()

    def PredictRevenue(self, request, context):
        # -------------------------------------------------
        # STEP 1: Receive request and update Lamport clock
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

        # -------------------------------------------------
        # STEP 2: Perform movie revenue prediction
        # -------------------------------------------------
        predicted_revenue = request.budget * 3.2

        print(
            "Predicted Revenue:",
            predicted_revenue
        )
        # -------------------------------------------------
        # STEP 3: Increment clock before sending response
        # -------------------------------------------------
        response_time = self.clock.increment()

        print(
            "Response Lamport Timestamp:",
            response_time
        )
        # -------------------------------------------------
        # STEP 4: Send response with Lamport timestamp
        # -------------------------------------------------
        return movie_pb2.PredictionResponse(
            predicted_revenue=predicted_revenue,
            message="Prediction completed successfully!",
            lamport_timestamp=response_time
        )

def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    movie_pb2_grpc.add_MoviePredictionServiceServicer_to_server(
        MoviePredictionService(),
        server
    )

    server.add_insecure_port('[::]:50051')

    server.start()

    print("Prediction Server is running on port 50051...")
    print("Lamport Clock Synchronization Enabled")

    server.wait_for_termination()


if __name__ == "__main__":
    serve()