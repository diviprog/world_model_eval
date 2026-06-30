"""Failure-mode diagnostic layer for robot manipulation policies.

Reads JSONL rollout records and produces an actionable failure breakdown. It
depends on nothing but pandas and the standard library, and never imports
SimplerEnv. See ARCHITECTURE.md for the seam that makes this possible.
"""

from diagnostics.schema import RolloutRecord, load_jsonl, validate

__all__ = ["RolloutRecord", "load_jsonl", "validate"]
