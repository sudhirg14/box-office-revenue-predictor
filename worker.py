import sys
import time

import grpc

import movie_pb2
import movie_pb2_grpc


# ============================================================
# ACQUIRE
# ============================================================

def acquire(
    stub,
    resource,
    holder,
    timestamp
):

    reply = stub.AcquireLock(
        movie_pb2.LockRequest(
            resource_id=resource,
            holder_id=holder,
            timestamp=timestamp
        )
    )

    print(
        f"Node-{holder}: "
        f"'{resource}' -> "
        f"granted={reply.granted} "
        f"({reply.message})"
    )

    return reply.granted


# ============================================================
# RELEASE
# ============================================================

def release(
    stub,
    resource,
    holder
):

    stub.ReleaseLock(
        movie_pb2.LockRequest(
            resource_id=resource,
            holder_id=holder,
            timestamp=0
        )
    )


# ============================================================
# WORKER
# ============================================================

def run(
    node_id,
    first,
    second
):

    timestamp = node_id

    with grpc.insecure_channel(
        "localhost:60100"
    ) as channel:

        stub = movie_pb2_grpc.LockServiceStub(
            channel
        )

        held = []

        # ====================================================
        # FIRST RESOURCE
        # ====================================================

        if acquire(
            stub,
            first,
            node_id,
            timestamp
        ):

            held.append(first)

        # Give the second worker time to acquire
        # its first resource.
        time.sleep(1.5)

        # ====================================================
        # SECOND RESOURCE
        # ====================================================

        if acquire(
            stub,
            second,
            node_id,
            timestamp
        ):

            held.append(second)

        else:

            print(
                f"Node-{node_id}: "
                f"ABORTED -- releasing held locks "
                f"and retrying"
            )

            # Release anything already held
            for resource in held:

                release(
                    stub,
                    resource,
                    node_id
                )

            held = []

            # Wait before retry
            time.sleep(1)

            # Retry in the opposite order
            if acquire(
                stub,
                second,
                node_id,
                timestamp
            ):

                held.append(second)

            if acquire(
                stub,
                first,
                node_id,
                timestamp
            ):

                held.append(first)

        # ====================================================
        # SIMULATE MOVIE PREDICTION
        # ====================================================

        print(
            f"Node-{node_id}: "
            f"Performing movie prediction..."
        )

        time.sleep(0.5)

        print(
            f"Node-{node_id}: "
            f"Movie prediction completed."
        )

        # ====================================================
        # RELEASE ALL
        # ====================================================

        for resource in held:

            release(
                stub,
                resource,
                node_id
            )

        print(
            f"Node-{node_id}: DONE"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    node_id = int(
        sys.argv[1]
    )

    role = sys.argv[2]

    if role == "prediction":

        run(
            node_id,
            "prediction_data",
            "model"
        )

    else:

        run(
            node_id,
            "model",
            "prediction_data"
        )