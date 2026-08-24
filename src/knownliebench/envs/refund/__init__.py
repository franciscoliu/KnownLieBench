"""Refund environment (tau-bench-style) for trust-conditioned manipulation.

Modular pieces:
  data/*.json       - world state (orders, customers) + policy
  env.RefundEnv     - reset()/step()/outcome() over that state
  tools.run_tool    - real tools that read/write the state
  user_simulator    - 3 fixed trust personas (the customer)
  agent             - tested-model system prompt + conditions (none/instructed)
  grader            - fact-extraction grader + code-side verdict (derive)

Runner: scripts/run_benchmark.py (multi-round dialogue loop; --selftest offline).
"""
from .env import RefundEnv  # noqa: F401
