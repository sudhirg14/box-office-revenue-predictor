from concurrent import futures
import threading
import time

import grpc

import movie_pb2
import movie_pb2_grpc

from lamport_clock import LamportClock


# =========================================================
# BACKEND SERVERS
# =========================================================

BACKENDS = [

    "localhost:50051",
    "localhost:50052",
    "localhost:50053"

]


# =========================================================
# ACTIVE CONNECTION COUNTERS
# =========================================================

active_connections = [
    0 for _ in BACKENDS
]

lock = threading.Lock()


# =========================================================
# LOAD BALANCER
# =========================================================

class LoadBalancer(
    movie_pb2_grpc.MoviePredictionServiceServicer
):

    def __init__(self):

        self.clock = LamportClock()

        print("\n========================================")
        print(
            "LEAST CONNECTIONS LOAD BALANCER"
        )
        print("========================================")

        print(
            "Load Balancer Port: 50050"
        )

        print(
            "Backend Servers:"
        )

        for backend in BACKENDS:

            print(
                f"  - {backend}"
            )

        print("========================================\n")

    # =====================================================
    # LEAST CONNECTIONS ALGORITHM
    # =====================================================

    def pick_least_connections(self):

        with lock:

            # Find minimum active connections

            minimum = min(
                active_connections
            )

            # Find backend having minimum

            index = active_connections.index(
                minimum
            )

            # IMPORTANT:
            # Increment BEFORE dispatch.
            #
            # This prevents another concurrent
            # request from seeing this backend
            # as still free.

            active_connections[index] += 1

            print("\n----------------------------------------")

            print(
                "[LB] LEAST CONNECTIONS DECISION"
            )

            print(
                "[LB] Selected Backend:",
                BACKENDS[index]
            )

            print(
                "[LB] Active Connections:",
                active_connections
            )

            print("----------------------------------------")

            return index

    # =====================================================
    # RELEASE CONNECTION
    # =====================================================

    def release_connection(
        self,
        index
    ):

        with lock:

            active_connections[index] -= 1

            print("\n----------------------------------------")

            print(
                "[LB] CONNECTION RELEASED"
            )

            print(
                "[LB] Backend:",
                BACKENDS[index]
            )

            print(
                "[LB] Active Connections:",
                active_connections
            )

            print("----------------------------------------")

    # =====================================================
    # gRPC PREDICTION
    # =====================================================

    def PredictRevenue(
        self,
        request,
        context
    ):

        # -------------------------------------------------
        # Receive request
        # -------------------------------------------------

        lb_time = self.clock.update(
            request.lamport_timestamp
        )

        print("\n========================================")
        print(
            "[LB] INCOMING PREDICTION REQUEST"
        )
        print("========================================")

        print(
            "[LB] Client Lamport:",
            request.lamport_timestamp
        )

        print(
            "[LB] Load Balancer Lamport:",
            lb_time
        )

        # -------------------------------------------------
        # Select least-loaded backend
        # -------------------------------------------------

        backend_index = (
            self.pick_least_connections()
        )

        backend_address = (
            BACKENDS[backend_index]
        )

        try:

            # -------------------------------------------------
            # Connect to selected backend
            # -------------------------------------------------

            with grpc.insecure_channel(
                backend_address
            ) as channel:

                stub = (
                    movie_pb2_grpc
                    .MoviePredictionServiceStub(
                        channel
                    )
                )

                print(
                    f"[LB] Forwarding request "
                    f"to {backend_address}"
                )

                # -------------------------------------------------
                # Forward ACTUAL MovieRequest
                # -------------------------------------------------

                response = stub.PredictRevenue(

                    request,

                    timeout=15

                )

            # -------------------------------------------------
            # Receive backend response
            # -------------------------------------------------

            response_time = (
                self.clock.update(
                    response.lamport_timestamp
                )
            )

            print(
                f"[LB] Response received "
                f"from {backend_address}"
            )

            print(
                "[LB] Predicted Revenue:",
                response.predicted_revenue,
                "million"
            )

            print(
                "[LB] Backend Lamport:",
                response.lamport_timestamp
            )

            print(
                "[LB] Updated LB Lamport:",
                response_time
            )

            # -------------------------------------------------
            # Return response to original client
            # -------------------------------------------------

            return movie_pb2.PredictionResponse(

                predicted_revenue=
                    response.predicted_revenue,

                message=(
                    f"{response.message} "
                    f"| Routed by Least Connections "
                    f"to {backend_address}"
                ),

                lamport_timestamp=
                    self.clock.increment()
            )

        except grpc.RpcError as e:

            print(
                f"[LB] ERROR: Backend "
                f"{backend_address} unavailable"
            )

            print(
                f"[LB] gRPC Error: {e}"
            )

            context.set_code(
                grpc.StatusCode.UNAVAILABLE
            )

            context.set_details(
                f"Backend unavailable: "
                f"{backend_address}"
            )

            return movie_pb2.PredictionResponse(

                predicted_revenue=0.0,

                message=(
                    f"Backend unavailable: "
                    f"{backend_address}"
                ),

                lamport_timestamp=
                    self.clock.increment()
            )

        finally:

            # -------------------------------------------------
            # ALWAYS release connection count
            # -------------------------------------------------

            self.release_connection(
                backend_index
            )


# =========================================================
# START LOAD BALANCER
# =========================================================

def serve():

    server = grpc.server(

        futures.ThreadPoolExecutor(
            max_workers=20
        )

    )

    movie_pb2_grpc.add_MoviePredictionServiceServicer_to_server(

        LoadBalancer(),

        server

    )

    server.add_insecure_port(
        "[::]:50050"
    )

    server.start()

    print(
        "[LB] Load Balancer started successfully."
    )

    print(
        "[LB] Listening on localhost:50050"
    )

    print(
        "[LB] Strategy: Least Connections"
    )

    print(
        "[LB] Backend count:",
        len(BACKENDS)
    )

    try:

        while True:

            time.sleep(86400)

    except KeyboardInterrupt:

        print(
            "\n[LB] Shutting down..."
        )

        server.stop(0)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    serve()