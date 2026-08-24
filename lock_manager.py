import sys
import threading
import time

from concurrent import futures

import grpc

import movie_pb2
import movie_pb2_grpc


# ============================================================
# DEADLOCK DETECTION MODE
# ============================================================

DETECT = (
    len(sys.argv) > 1
    and sys.argv[1] == "detect"
)


# ============================================================
# LOCK MANAGER
# ============================================================

class LockManager(
    movie_pb2_grpc.LockServiceServicer
):

    def __init__(self):

        # Protect shared lock-manager state
        self.mutex = threading.Lock()

        # Two shared resources
        #
        # prediction_data
        # model
        #
        # None means resource is free

        self.locks = {
            "prediction_data": None,
            "model": None
        }

        # holder_id -> resource it is waiting for
        self.wait_for = {}

        # Conditions allow blocked workers to wait
        self.conditions = {
            "prediction_data": threading.Condition(),
            "model": threading.Condition()
        }

    # ========================================================
    # DEADLOCK DETECTION
    # ========================================================

    def _would_cycle(self, holder, resource):

        """
        Check whether allowing holder to wait for
        resource would create a cycle.

        Start from the current owner of the resource
        and follow the wait-for chain.

        If we eventually reach 'holder',
        a circular wait exists.
        """

        visited = set()

        current_resource = resource

        while True:

            owner = self.locks.get(
                current_resource
            )

            # Resource is free
            if owner is None:

                return False

            # Owner is the same process requesting it
            if owner == holder:

                return True

            # Already visited
            if owner in visited:

                return False

            visited.add(owner)

            # What resource is that owner waiting for?
            current_resource = self.wait_for.get(
                owner
            )

            if current_resource is None:

                return False

    # ========================================================
    # ACQUIRE LOCK
    # ========================================================

    def AcquireLock(
        self,
        request,
        context
    ):

        resource = request.resource_id

        holder = request.holder_id

        timestamp = request.timestamp

        cond = self.conditions[resource]

        with cond:

            with self.mutex:

                owner = self.locks.get(
                    resource
                )

                # ------------------------------------------------
                # RESOURCE IS FREE
                # ------------------------------------------------

                if owner is None:

                    self.locks[resource] = holder

                    self.wait_for.pop(
                        holder,
                        None
                    )

                    print(
                        f"[LockManager] "
                        f"Node-{holder} ACQUIRED "
                        f"'{resource}'"
                    )

                    return movie_pb2.LockReply(
                        granted=True,
                        message="granted"
                    )

                # ------------------------------------------------
                # DEADLOCK DETECTION
                # ------------------------------------------------

                if DETECT and self._would_cycle(
                    holder,
                    resource
                ):

                    print()

                    print(
                        f"[LockManager] "
                        f"DEADLOCK DETECTED!"
                    )

                    print(
                        f"[LockManager] "
                        f"Node-{holder} -> "
                        f"'{resource}' "
                        f"(held by Node-{owner}) "
                        f"would close a cycle."
                    )

                    print(
                        f"[LockManager] "
                        f"ABORTING Node-{holder}"
                    )

                    return movie_pb2.LockReply(
                        granted=False,
                        message="deadlock-abort"
                    )

                # ------------------------------------------------
                # RESOURCE BUSY
                # ------------------------------------------------

                self.wait_for[holder] = resource

                print(
                    f"[LockManager] "
                    f"Node-{holder} WAITING for "
                    f"'{resource}' "
                    f"(held by Node-{owner})"
                )

            # ----------------------------------------------------
            # WAIT
            # ----------------------------------------------------

            while True:

                with self.mutex:

                    owner = self.locks.get(
                        resource
                    )

                    if owner is None:

                        self.locks[resource] = holder

                        self.wait_for.pop(
                            holder,
                            None
                        )

                        print(
                            f"[LockManager] "
                            f"Node-{holder} ACQUIRED "
                            f"'{resource}' "
                            f"after waiting"
                        )

                        return movie_pb2.LockReply(
                            granted=True,
                            message="granted-after-wait"
                        )

                cond.wait(
                    timeout=1
                )

    # ========================================================
    # RELEASE LOCK
    # ========================================================

    def ReleaseLock(
        self,
        request,
        context
    ):

        resource = request.resource_id

        holder = request.holder_id

        cond = self.conditions[resource]

        with cond:

            with self.mutex:

                if self.locks.get(
                    resource
                ) == holder:

                    self.locks[resource] = None

                    print(
                        f"[LockManager] "
                        f"Node-{holder} RELEASED "
                        f"'{resource}'"
                    )

                    cond.notify_all()

                    return movie_pb2.LockReply(
                        granted=True,
                        message="released"
                    )

                return movie_pb2.LockReply(
                    granted=False,
                    message="not-owner"
                )


# ============================================================
# START SERVER
# ============================================================

def serve():

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=10
        )
    )

    movie_pb2_grpc.add_LockServiceServicer_to_server(
        LockManager(),
        server
    )

    server.add_insecure_port(
        "localhost:60100"
    )

    server.start()

    print(
        f"LockManager started on localhost:60100 "
        f"(deadlock detection = {DETECT})"
    )

    try:

        while True:

            time.sleep(86400)

    except KeyboardInterrupt:

        print(
            "LockManager shutting down..."
        )

        server.stop(0)


if __name__ == "__main__":

    serve()