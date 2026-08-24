"""KnownLieBench: a knowledge-verified benchmark of emergent deception in LLM agents.

The benchmark runs multi-round customer-service dialogues in eight legally grounded
domains. Each round first verifies, through a neutral probe, that the agent knows the
customer's entitlement, and only then tests whether the agent falsely denies that same
entitlement under an incentive or an explicit instruction.

Entry points live in ``scripts/``:

    python scripts/run_benchmark.py --help     # run dialogues
    python scripts/score_dialogue.py --help    # score a finished run
"""

__version__ = "1.0.0"
