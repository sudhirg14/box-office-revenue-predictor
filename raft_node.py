import sys
import time
import random
import threading

from concurrent import futures

import grpc

import raft_pb2
import raft_pb2_grpc


# ============================================================
# RAFT CLUSTER CONFIGURATION
# ============================================================

ALL_NODES = {
    "A": "localhost:60401",
    "B": "localhost:60402",
    "C": "localhost:60403",
}


HEARTBEAT_INTERVAL = 0.5
ELECTION_TIMEOUT_RANGE = (1.5, 3.0)


# ============================================================
# RAFT NODE
# ============================================================

class RaftNode(raft_pb2_grpc.RaftServiceServicer):

    def __init__(self, node_id):

        self.id = node_id

        self.peers = {
            node: address
            for node, address in ALL_NODES.items()
            if node != node_id
        }

        self.lock = threading.Lock()

        # Raft state
        self.state = "FOLLOWER"

        self.current_term = 0

        self.voted_for = None

        self.leader_id = None

        self.last_heartbeat = time.time()


    def is_leader(self):

        with self.lock:

            return self.state == "LEADER"


    def get_leader_id(self):

        with self.lock:

            return self.leader_id


    # ========================================================
    # LOGGING
    # ========================================================

    def log(self, message):

        print(
            f"[{self.id} "
            f"term={self.current_term} "
            f"{self.state}] "
            f"{message}",
            flush=True
        )


    # ========================================================
    # ELECTION TIMER
    # ========================================================

    def _reset_election_timer(self):

        self.last_heartbeat = time.time()


    def _random_timeout(self):

        return random.uniform(
            *ELECTION_TIMEOUT_RANGE
        )


    # ========================================================
    # REQUEST VOTE RPC
    # ========================================================

    def RequestVote(self, request, context):

        with self.lock:

            # Higher term means newer election
            if request.term > self.current_term:

                self.current_term = request.term

                self.voted_for = None

                self.state = "FOLLOWER"


            grant = (
                request.term >= self.current_term
                and
                self.voted_for in (
                    None,
                    request.candidate_id
                )
            )


            if grant:

                self.voted_for = request.candidate_id

                self._reset_election_timer()

                self.log(
                    f"Voted for "
                    f"{request.candidate_id} "
                    f"(term {request.term})"
                )


            return raft_pb2.VoteReply(
                term=self.current_term,
                vote_granted=grant
            )


    # ========================================================
    # APPEND ENTRIES / HEARTBEAT RPC
    # ========================================================

    def AppendEntries(self, request, context):

        with self.lock:

            if request.term >= self.current_term:

                self.current_term = request.term

                self.state = "FOLLOWER"

                self.leader_id = request.leader_id

                self.log(
                    f"Received heartbeat from leader "
                    f"{request.leader_id}"
                )

                self._reset_election_timer()

                return raft_pb2.HeartbeatReply(
                    term=self.current_term,
                    success=True
                )


            return raft_pb2.HeartbeatReply(
                term=self.current_term,
                success=False
            )


    # ========================================================
    # ELECTION WATCHDOG
    # ========================================================

    def _election_watchdog(self):

        while True:

            timeout = self._random_timeout()

            time.sleep(0.1)

            elapsed = (
                time.time()
                - self.last_heartbeat
            )

            while elapsed < timeout:

                time.sleep(0.1)

                if self.state == "LEADER":

                    break

                elapsed = (
                    time.time()
                    - self.last_heartbeat
                )


            if (
                self.state != "LEADER"
                and
                elapsed >= timeout
            ):

                self._start_election()


    # ========================================================
    # START ELECTION
    # ========================================================

    def _start_election(self):

        with self.lock:

            self.state = "CANDIDATE"

            self.current_term += 1

            self.voted_for = self.id

            term = self.current_term

            self._reset_election_timer()

            self.log("Starting election")


        votes = 1


        # Request votes from other nodes
        for peer_id, address in self.peers.items():

            try:

                with grpc.insecure_channel(address) as channel:

                    stub = (
                        raft_pb2_grpc
                        .RaftServiceStub(channel)
                    )


                    reply = stub.RequestVote(

                        raft_pb2.VoteRequest(

                            term=term,

                            candidate_id=self.id
                        ),

                        timeout=1
                    )


                    if reply.vote_granted:

                        votes += 1


                    elif reply.term > term:

                        with self.lock:

                            self.current_term = (
                                reply.term
                            )

                            self.state = "FOLLOWER"

                        return


            except grpc.RpcError:

                self.log(
                    f"No response from "
                    f"{peer_id} "
                    f"(may be down)"
                )


        # ====================================================
        # CHECK MAJORITY
        # ====================================================

        with self.lock:

            if (
                self.state == "CANDIDATE"
                and
                votes > len(ALL_NODES) // 2
            ):

                self.state = "LEADER"

                self.leader_id = self.id

                self.log(
                    f"*** ELECTED LEADER *** "
                    f"({votes}/{len(ALL_NODES)} votes)"
                )

            else:

                self.state = "FOLLOWER"

                self.log(
                    f"Election failed "
                    f"({votes}/{len(ALL_NODES)} votes) "
                    f"- reverting to follower"
                )


    # ========================================================
    # HEARTBEAT LOOP
    # ========================================================

    def _heartbeat_loop(self):

        while True:

            time.sleep(
                HEARTBEAT_INTERVAL
            )


            if self.state != "LEADER":

                continue


            for peer_id, address in self.peers.items():

                try:

                    with grpc.insecure_channel(
                        address
                    ) as channel:

                        stub = (
                            raft_pb2_grpc
                            .RaftServiceStub(channel)
                        )


                        stub.AppendEntries(

                            raft_pb2.HeartbeatRequest(

                                term=self.current_term,

                                leader_id=self.id
                            ),

                            timeout=1
                        )


                except grpc.RpcError:

                    pass


# ============================================================
# START RAFT SERVER
# ============================================================

def serve(node_id):

    if node_id not in ALL_NODES:

        print(
            "Invalid node ID."
        )

        print(
            "Use A, B, or C."
        )

        return


    node = RaftNode(node_id)


    server = grpc.server(

        futures.ThreadPoolExecutor(
            max_workers=20
        )
    )


    raft_pb2_grpc.add_RaftServiceServicer_to_server(

        node,

        server
    )


    port = ALL_NODES[node_id].split(":")[1]


    server.add_insecure_port(
        f"localhost:{port}"
    )


    server.start()


    print(
        f"RaftNode-{node_id} "
        f"listening on "
        f"{ALL_NODES[node_id]}"
    )

        # Start election watchdog
    threading.Thread(
        target=node._election_watchdog,
        daemon=True
    ).start()

    # Start heartbeat loop
    threading.Thread(
        target=node._heartbeat_loop,
        daemon=True
    ).start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print(
            f"\nRaftNode-{node_id} shutting down..."
        )
        server.stop(0)


    # Election watchdog
    threading.Thread(

        target=node._election_watchdog,

        daemon=True

    ).start()


    # Heartbeat thread
    threading.Thread(

        target=node._heartbeat_loop,

        daemon=True

    ).start()


    try:

        while True:

            time.sleep(86400)


    except KeyboardInterrupt:

        print(
            f"RaftNode-{node_id} "
            f"shutting down..."
        )

        server.stop(0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "python raft_node.py A|B|C"
        )

        sys.exit(1)


    serve(sys.argv[1])