import time
import threading
import grpc

import replica_pb2
import replica_pb2_grpc


REPLICAS = {
    "A": "localhost:60301",
    "B": "localhost:60302",
    "C": "localhost:60303",
}


def save(replica_name, key, content):
    """Send a write request to a specific replica."""

    with grpc.insecure_channel(REPLICAS[replica_name]) as channel:

        stub = replica_pb2_grpc.ReplicaServiceStub(channel)

        ack = stub.SaveValue(
            replica_pb2.ValueUpdate(
                key=key,
                content=content,
                lamport_timestamp=0,
                origin_replica=""
            )
        )

        print(
            f'[Client] Saved on Replica-{replica_name}: '
            f'"{content}" '
            f'-> accepted={ack.accepted}, '
            f'Lamport timestamp={ack.lamport_timestamp}'
        )


def read(replica_name, key):
    """Read the current value from a replica."""

    with grpc.insecure_channel(REPLICAS[replica_name]) as channel:

        stub = replica_pb2_grpc.ReplicaServiceStub(channel)

        state = stub.GetValue(
            replica_pb2.ValueQuery(
                key=key
            )
        )

        return (
            state.content,
            state.lamport_timestamp,
            state.origin_replica
        )


def main():

    key = "shared-key-1"

    print(
        "\n=== Experiment 7: Eventual Consistency "
        "via Gossip Replication + LWW ===\n"
    )

    print(
        "Simulating two conflicting writers "
        "hitting different replicas...\n"
    )

    # Writer 1 → Replica A
    t1 = threading.Thread(
        target=save,
        args=(
            "A",
            key,
            "value-from-writer-1"
        )
    )

    # Writer 2 → Replica C
    t2 = threading.Thread(
        target=save,
        args=(
            "C",
            key,
            "value-from-writer-2"
        )
    )

    # Start both writers
    t1.start()

    time.sleep(0.05)

    t2.start()

    # Wait for both local writes to finish
    t1.join()
    t2.join()

    # ---------------------------------------------------------
    # Check state immediately after writes
    # ---------------------------------------------------------

    print(
        "\n=== Immediately after writes ==="
    )

    for name in REPLICAS:

        try:

            content, timestamp, origin = read(
                name,
                key
            )

            print(
                f'Replica-{name}: '
                f'"{content}" '
                f'(ts={timestamp}, origin={origin})'
            )

        except grpc.RpcError:

            print(
                f"Replica-{name}: unavailable"
            )

    # ---------------------------------------------------------
    # Wait for gossip
    # ---------------------------------------------------------

    print(
        "\nWaiting 2 seconds for gossip "
        "replication...\n"
    )

    time.sleep(2)

    # ---------------------------------------------------------
    # Check final state
    # ---------------------------------------------------------

    print(
        "=== After convergence window ==="
    )

    results = {}

    for name in REPLICAS:

        try:

            content, timestamp, origin = read(
                name,
                key
            )

            results[name] = content

            print(
                f'Replica-{name}: '
                f'"{content}" '
                f'(ts={timestamp}, origin={origin})'
            )

        except grpc.RpcError:

            print(
                f"Replica-{name}: unavailable"
            )

    # ---------------------------------------------------------
    # Verify convergence
    # ---------------------------------------------------------

    if len(results) == 3 and len(set(results.values())) == 1:

        final_value = list(
            results.values()
        )[0]

        print(
            "\n*** CONVERGED ***"
        )

        print(
            f'All replicas agree on: "{final_value}"'
        )

    else:

        print(
            "\n*** NOT YET CONVERGED ***"
        )

        print(
            f"Replica values: {results}"
        )


if __name__ == "__main__":
    main()