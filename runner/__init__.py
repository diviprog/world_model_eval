"""SimplerEnv-facing harness: runs rollouts and writes schema records.

Depends on the simulator and a GPU. The diagnostics package never imports from
here — the two halves communicate only through JSONL records on disk.
"""
