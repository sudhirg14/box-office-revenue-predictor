import sys
import time
import threading
from concurrent import futures

import grpc

import movie_pb2
import movie_pb2_grpc
from lamport_clock import LamportClock


# ============================================================
# THREE DISTRIBUTED MOVIE PREDICTION NODES
# ============================================================

PEERS = {
    1: "localhost:50051",
    2: "localhost:50052",
    3: "localhost:50053"
}


# ============================================================
# NODE
# ============================================================

class Node:

    def __init__(self, node_id):

        self.id = node_id

        # All nodes except myself
        self.peers = {
            node_id_: address
            for node_id_, address in PEERS.items()
            if node_id_ != node_id
        }

        # Existing Lamport Clock from Exp 3
        self.clock = LamportClock()

        # Ricart-Agrawala states
        self.state = "RELEASED"

        self.request_time = None

        # Thread synchronization
        self.lock = threading.Lock()

        # Requests that must be deferred
        self.deferred = []

    # ========================================================
    # LAMPORT CLOCK
    # ========================================================

    def tick(self):

        return self.clock.increment()

    def update(self, received_timestamp):

        return self.clock.update(received_timestamp)

    # ========================================================
    # START gRPC SERVER
    # ========================================================

    def serve(self):

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=20)
        )

        # Register mutual exclusion service
        movie_pb2_grpc.add_MutexServiceServicer_to_server(
            MutexServicer(self),
            server
        )

        server.add_insecure_port(
            PEERS[self.id]
        )

        server.start()

        print(
            f"Node-{self.id}: Server started on "
            f"{PEERS[self.id]}"
        )

        return server

    # ========================================================
    # SEND REQUEST TO PEER
    # ========================================================

    def request_permission(self, address, timestamp):

        deadline = time.time() + 30

        while time.time() < deadline:

            try:

                with grpc.insecure_channel(address) as channel:

                    stub = movie_pb2_grpc.MutexServiceStub(
                        channel
                    )

                    response = stub.RequestAccess(
                        movie_pb2.AccessRequest(
                            node_id=self.id,
                            timestamp=timestamp
                        )
                    )

                    return response

            except grpc.RpcError:

                print(
                    f"Node-{self.id}: Waiting for peer "
                    f"{address}..."
                )

                time.sleep(0.5)

        raise RuntimeError(
            f"Could not connect to {address}"
        )

    # ========================================================
    # REQUEST CRITICAL SECTION
    # ========================================================

    def request_cs(self):

        # ----------------------------------------------------
        # STEP 1: Generate Lamport timestamp
        # ----------------------------------------------------

        my_timestamp = self.tick()

        with self.lock:

            self.state = "WANTED"

            self.request_time = my_timestamp

        print()
        print("=" * 60)
        print(
            f"Node-{self.id}: WANTS critical section"
        )
        print(
            f"Node-{self.id}: Request timestamp = "
            f"{my_timestamp}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # STEP 2: Ask every other node for permission
        # ----------------------------------------------------

        for peer_id, address in self.peers.items():

            print(
                f"Node-{self.id}: Requesting permission "
                f"from Node-{peer_id}"
            )

            response = self.request_permission(
                address,
                my_timestamp
            )

            # Update Lamport clock
            new_time = self.update(
                response.timestamp
            )

            print(
                f"Node-{self.id}: Permission received "
                f"from Node-{peer_id}"
            )

            print(
                f"Node-{self.id}: Updated Lamport clock = "
                f"{new_time}"
            )

        # ----------------------------------------------------
        # STEP 3: Enter critical section
        # ----------------------------------------------------

        with self.lock:

            self.state = "HELD"

        print()
        print("*" * 60)
        print(
            f"Node-{self.id}: ENTERED CRITICAL SECTION"
        )
        print(
            f"Node-{self.id}: Performing movie "
            f"box-office prediction..."
        )
        print("*" * 60)

        # ----------------------------------------------------
        # Simulate prediction operation
        # ----------------------------------------------------

        budget = 100_000_000

        predicted_revenue = budget * 3.2

        print(
            f"Node-{self.id}: Budget = ₹{budget:,}"
        )

        print(
            f"Node-{self.id}: Predicted Revenue = "
            f"₹{predicted_revenue:,.2f}"
        )

        # Keep node inside critical section long enough
        # for contention to be visible
        time.sleep(3)

        # ----------------------------------------------------
        # STEP 4: Release critical section
        # ----------------------------------------------------

        self.release_cs()

    # ========================================================
    # RELEASE CRITICAL SECTION
    # ========================================================

    def release_cs(self):

        with self.lock:

            self.state = "RELEASED"

            deferred_requests = self.deferred

            self.deferred = []

        print()
        print("-" * 60)
        print(
            f"Node-{self.id}: RELEASED CRITICAL SECTION"
        )
        print("-" * 60)

        # Allow deferred RPC calls to return
        for event in deferred_requests:

            event.set()


# ============================================================
# MUTEX gRPC SERVICE
# ============================================================

class MutexServicer(
    movie_pb2_grpc.MutexServiceServicer
):

    def __init__(self, node):

        self.node = node

    # ========================================================
    # RequestAccess
    # ========================================================

    def RequestAccess(self, request, context):

        node = self.node

        # ----------------------------------------------------
        # Update Lamport clock when request is received
        # ----------------------------------------------------

        received_time = node.update(
            request.timestamp
        )

        print()
        print(
            f"Node-{node.id}: Received request from "
            f"Node-{request.node_id}"
        )

        print(
            f"Node-{node.id}: Request timestamp = "
            f"{request.timestamp}"
        )

        print(
            f"Node-{node.id}: Local Lamport clock = "
            f"{received_time}"
        )

        # ----------------------------------------------------
        # Ricart-Agrawala decision
        # ----------------------------------------------------

        with node.lock:

            defer = (

                node.state == "HELD"

                or (

                    node.state == "WANTED"

                    and
                    (
                        node.request_time,
                        node.id
                    )
                    <
                    (
                        request.timestamp,
                        request.node_id
                    )
                )
            )

        # ----------------------------------------------------
        # DEFER REQUEST
        # ----------------------------------------------------

        if defer:

            print(
                f"Node-{node.id}: DEFERS request from "
                f"Node-{request.node_id}"
            )

            event = threading.Event()

            with node.lock:

                node.deferred.append(event)

            # Block the gRPC response
            event.wait()

        # ----------------------------------------------------
        # GRANT REQUEST
        # ----------------------------------------------------

        else:

            print(
                f"Node-{node.id}: GRANTS request to "
                f"Node-{request.node_id}"
            )

        # ----------------------------------------------------
        # Send reply timestamp
        # ----------------------------------------------------

        send_timestamp = node.tick()

        return movie_pb2.AccessReply(
            node_id=node.id,
            timestamp=send_timestamp
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage: python node.py <node_id>"
        )

        print(
            "Example: python node.py 1"
        )

        sys.exit(1)

    node_id = int(sys.argv[1])

    if node_id not in PEERS:

        print(
            "Node ID must be 1, 2, or 3"
        )

        sys.exit(1)

    node = Node(node_id)

    server = node.serve()

    # --------------------------------------------------------
    # Allow all three nodes to start
    # --------------------------------------------------------

    print(
        f"Node-{node_id}: Waiting for other nodes..."
    )

    time.sleep(6)

    # --------------------------------------------------------
    # Start Ricart-Agrawala algorithm
    # --------------------------------------------------------

    node.request_cs()

    # --------------------------------------------------------
    # Keep server alive briefly
    # --------------------------------------------------------

    time.sleep(2)

    server.stop(0)

    print(
        f"Node-{node_id}: Shutdown complete"
    )


if __name__ == "__main__":

    main()