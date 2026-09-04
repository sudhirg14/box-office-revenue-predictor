import sys
import time
import threading
from concurrent import futures

import grpc
import replica_pb2
import replica_pb2_grpc


# Addresses of all replicas
ALL_REPLICAS = {
    "A": "localhost:60301",
    "B": "localhost:60302",
    "C": "localhost:60303",
}


class ReplicaNode(replica_pb2_grpc.ReplicaServiceServicer):

    def __init__(self, name):
        self.name = name

        # Every replica knows about the other replicas
        self.peers = {
            n: addr
            for n, addr in ALL_REPLICAS.items()
            if n != name
        }

        # Lamport logical clock
        self.clock = 0

        # Protect shared data
        self.lock = threading.Lock()

        # key -> (content, timestamp, origin)
        self.store = {}

    def log(self, message):
        print(
            f"[Replica-{self.name} | Lamport={self.clock}] {message}",
            flush=True
        )

    # ---------------------------------------------------------
    # Lamport clock functions
    # ---------------------------------------------------------

    def tick(self):
        """Increment clock for a local event."""
        with self.lock:
            self.clock += 1
            return self.clock

    def update_clock(self, received_timestamp):
        """Update clock when receiving a message."""
        with self.lock:
            self.clock = max(
                self.clock,
                received_timestamp
            ) + 1

            return self.clock

    # ---------------------------------------------------------
    # Last-Write-Wins
    # ---------------------------------------------------------

    def _apply_if_newer(self, key, content, timestamp, origin):
        """
        Apply the update only if it is newer.

        LWW comparison:
            (timestamp, origin)

        Origin is used as a deterministic tie-breaker
        when two timestamps are equal.
        """

        with self.lock:

            current = self.store.get(key)

            if current is None:

                self.store[key] = (
                    content,
                    timestamp,
                    origin
                )

                self.log(
                    f'APPLIED "{key}" = "{content}" '
                    f'(ts={timestamp}, origin={origin})'
                )

                return True

            current_content, current_ts, current_origin = current

            if (timestamp, origin) > (
                current_ts,
                current_origin
            ):

                self.store[key] = (
                    content,
                    timestamp,
                    origin
                )

                self.log(
                    f'UPDATED "{key}" = "{content}" '
                    f'(ts={timestamp}, origin={origin})'
                )

                return True

            else:

                self.log(
                    f'IGNORED stale update for "{key}" '
                    f'(incoming={timestamp}/{origin}, '
                    f'current={current_ts}/{current_origin})'
                )

                return False

    # ---------------------------------------------------------
    # Gossip replication
    # ---------------------------------------------------------

    def _gossip(self, key, content, timestamp, origin):

        """
        Send the update to all peer replicas
        in the background.
        """

        for peer_name, address in self.peers.items():

            try:

                with grpc.insecure_channel(address) as channel:

                    stub = (
                        replica_pb2_grpc
                        .ReplicaServiceStub(channel)
                    )

                    stub.SyncUpdate(
                        replica_pb2.ValueUpdate(
                            key=key,
                            content=content,
                            lamport_timestamp=timestamp,
                            origin_replica=origin,
                        )
                    )

                    self.log(
                        f'Gossiped "{key}" -> Replica-{peer_name}'
                    )

            except grpc.RpcError as error:

                self.log(
                    f"Gossip to Replica-{peer_name} failed: "
                    f"{error.code()}"
                )

    # ---------------------------------------------------------
    # Client -> Replica
    # ---------------------------------------------------------

    def SaveValue(self, request, context):

        # 1. Generate Lamport timestamp
        timestamp = self.tick()

        # 2. Apply write locally FIRST
        self._apply_if_newer(
            request.key,
            request.content,
            timestamp,
            self.name
        )

        # 3. Start gossip in background
        threading.Thread(
            target=self._gossip,
            args=(
                request.key,
                request.content,
                timestamp,
                self.name,
            ),
            daemon=True,
        ).start()

        # 4. Immediately acknowledge client
        return replica_pb2.SaveAck(
            accepted=True,
            replica=self.name,
            lamport_timestamp=timestamp,
        )

    # ---------------------------------------------------------
    # Replica -> Replica
    # ---------------------------------------------------------

    def SyncUpdate(self, request, context):

        # Update local Lamport clock
        self.update_clock(
            request.lamport_timestamp
        )

        # Apply only if incoming update is newer
        self._apply_if_newer(
            request.key,
            request.content,
            request.lamport_timestamp,
            request.origin_replica,
        )

        return replica_pb2.SaveAck(
            accepted=True,
            replica=self.name,
            lamport_timestamp=self.clock,
        )

    # ---------------------------------------------------------
    # Read value
    # ---------------------------------------------------------

    def GetValue(self, request, context):

        with self.lock:
            entry = self.store.get(request.key)

        # Key does not exist
        if entry is None:

            return replica_pb2.ValueState(
                key=request.key,
                content="",
                lamport_timestamp=0,
                origin_replica="",
            )

        content, timestamp, origin = entry

        return replica_pb2.ValueState(
            key=request.key,
            content=content,
            lamport_timestamp=timestamp,
            origin_replica=origin,
        )


# =============================================================
# Start Replica Server
# =============================================================

def serve(name, port):

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=20
        )
    )

    replica_pb2_grpc.add_ReplicaServiceServicer_to_server(
        ReplicaNode(name),
        server
    )

    server.add_insecure_port(
        f"localhost:{port}"
    )

    server.start()

    print(
        f"Replica-{name} listening on "
        f"localhost:{port}",
        flush=True
    )

    try:

        while True:
            time.sleep(86400)

    except KeyboardInterrupt:

        server.stop(0)


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print(
            "Usage: python replica_node.py <A|B|C> <port>"
        )

        sys.exit(1)

    replica_name = sys.argv[1]
    port = int(sys.argv[2])

    serve(
        replica_name,
        port
    )