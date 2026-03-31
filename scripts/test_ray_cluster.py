# SPDX-License-Identifier: Apache-2.0
"""Ray cluster connectivity test for vllm-metal distributed inference.

Prerequisites:
  # Mac A (head node)
  RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --head --num-gpus=1

  # Mac B (worker node)
  RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 ray start --address=<Mac_A_IP>:6379 --num-gpus=1

Usage:
  python scripts/test_ray_cluster.py
"""

from __future__ import annotations

import argparse
import sys
import time


def test_cluster_resources(expected_nodes: int = 2) -> None:
    """Verify the Ray cluster has the expected GPU resources."""
    import ray

    resources = ray.cluster_resources()
    gpu_count = resources.get("GPU", 0)
    print(f"Cluster resources: {resources}")
    print(f"GPU count: {gpu_count}")

    if gpu_count < expected_nodes:
        print(
            f"FAIL: Expected {expected_nodes} GPUs, got {gpu_count}. "
            "Ensure all nodes started with --num-gpus=1."
        )
        sys.exit(1)

    nodes = ray.nodes()
    alive = [n for n in nodes if n["Alive"]]
    print(f"Alive nodes: {len(alive)}")
    for node in alive:
        addr = node.get("NodeManagerAddress", "unknown")
        res = node.get("Resources", {})
        print(f"  {addr}: GPU={res.get('GPU', 0)}")

    if len(alive) < expected_nodes:
        print(f"FAIL: Expected {expected_nodes} alive nodes, got {len(alive)}.")
        sys.exit(1)

    print("PASS: Cluster resources OK\n")


def test_ray_tensor_exchange() -> None:
    """Exchange tensors between nodes using pure Ray (no Gloo).

    This mirrors how the Ray compiled DAG transports intermediate
    tensors between PP stages — via Ray return values, not
    torch.distributed.
    """
    import ray
    import torch

    @ray.remote(num_gpus=1)
    class TensorWorker:
        """Ray actor that sends/receives tensors via Ray object store."""

        def get_node_ip(self) -> str:
            return ray.util.get_node_ip_address()

        def create_tensor(self) -> dict:
            """Create a test tensor and return it via Ray."""
            tensor = torch.arange(1024, dtype=torch.float32) + 1.0
            return {
                "hidden_states": tensor,
                "ip": ray.util.get_node_ip_address(),
            }

        def receive_and_verify(self, data: dict) -> dict:
            """Receive a tensor from another worker and verify it."""
            tensor = data["hidden_states"]
            expected = torch.arange(1024, dtype=torch.float32) + 1.0
            match = torch.equal(tensor, expected)
            return {
                "match": match,
                "shape": list(tensor.shape),
                "sender_ip": data["ip"],
                "receiver_ip": ray.util.get_node_ip_address(),
            }

    # Create workers — num_gpus=1 ensures one per node
    workers = [TensorWorker.remote() for _ in range(2)]

    # Verify they are on different nodes
    ips = ray.get([w.get_node_ip.remote() for w in workers])
    print(f"Worker IPs: {ips}")
    if len(set(ips)) < 2:
        print("WARNING: Both workers on same node. Cross-node test requires 2 nodes.")

    # Simulate PP: worker 0 creates tensor, worker 1 receives it
    print("Exchanging tensors via Ray object store...")
    start = time.monotonic()

    # Worker 0 creates IntermediateTensors-like data
    tensor_ref = workers[0].create_tensor.remote()

    # Worker 1 receives and verifies (cross-node transfer via Ray)
    result_ref = workers[1].receive_and_verify.remote(tensor_ref)
    result = ray.get(result_ref)

    elapsed = time.monotonic() - start

    print(f"Sender:   {result['sender_ip']}")
    print(f"Receiver: {result['receiver_ip']}")
    print(f"Shape:    {result['shape']}")
    print(f"Match:    {result['match']}")
    print(f"Time:     {elapsed:.3f}s")

    if not result["match"]:
        print("FAIL: Tensor data mismatch!")
        sys.exit(1)

    # Benchmark: larger tensor (simulate hidden states)
    # Reuse existing workers to avoid GPU resource contention.
    for _ in range(2):
        t0 = time.monotonic()
        ref = workers[0].create_tensor.remote()
        ray.get(workers[1].receive_and_verify.remote(ref))
        dt = time.monotonic() - t0
        bw = (1024 * 4) / dt / 1e6  # MB/s
        print(f"  round-trip (1024 floats): {dt:.3f}s, ~{bw:.0f} MB/s")

    print("PASS: Ray tensor exchange OK\n")


def main():
    parser = argparse.ArgumentParser(description="Test Ray cluster for vllm-metal")
    parser.add_argument("--nodes", type=int, default=2, help="Expected number of nodes")
    parser.add_argument(
        "--address",
        type=str,
        default="auto",
        help="Ray cluster address (default: auto)",
    )
    args = parser.parse_args()

    import ray

    ray.init(address=args.address)
    print(f"Connected to Ray cluster at {args.address}\n")

    test_cluster_resources(expected_nodes=args.nodes)
    test_ray_tensor_exchange()

    print("All tests passed!")
    ray.shutdown()


if __name__ == "__main__":
    main()
