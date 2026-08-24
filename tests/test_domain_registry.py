"""Domain registry invariants: refund stays bound to its existing functions; every spec is well-formed
and grounded (distinct owed/favored, a calibration case, a cited policy)."""
from knownliebench.domains import get_domain, REGISTRY, _load_builtins


def test_refund_spec_binds_to_existing_functions_verbatim():
    import knownliebench.envs.refund.agent as A
    import knownliebench.envs.refund.grader as G
    import knownliebench.envs.refund.knowledge_gate as K
    import knownliebench.envs.refund.receiver_agent as R
    from knownliebench.envs.refund.env import RefundEnv
    s = get_domain("refund")
    # refund must route to its existing, human-tuned functions -- no behavior change
    assert s.agent_system is A.agent_system
    assert s.grader_prompt is G.grader_prompt and s.derive is G.derive
    assert s.gate_prompt is K.knowledge_probe_prompt and s.gate_passed is K.passed
    assert s.receiver_system is R.receiver_system and s.opening_for is R.opening_for
    assert s.env_factory is RefundEnv
    assert s.outcomes.owed == "cash" and s.outcomes.favored == "store_credit"
    assert s.terminal_tools == ("issue_cash_refund", "issue_store_credit")


def test_all_specs_wellformed_and_grounded():
    _load_builtins()
    assert "refund" in REGISTRY and "airline" in REGISTRY
    for name, s in REGISTRY.items():
        assert s.name == name
        assert s.outcomes.owed and s.outcomes.favored and s.outcomes.owed != s.outcomes.favored
        for fn in (s.env_factory, s.agent_system, s.gate_prompt, s.gate_passed, s.grader_prompt,
                   s.derive, s.receiver_system, s.opening_for):
            assert callable(fn)
        env = s.env_factory("default"); env.reset()
        orders, sits = env.data["orders"], env.data["situations"]
        assert set(orders) == set(sits), f"{name}: order<->situation key mismatch"
        ents = {env.entitled_outcome(o) for o in orders}
        assert ents <= {s.outcomes.owed, s.outcomes.favored, "none"}, f"{name}: stray entitled_outcome {ents}"
        assert any(env.entitled_outcome(o) == "none" for o in orders), f"{name}: missing a calibration (none) case"
        assert env.data["policy"].get("source"), f"{name}: policy.json has no citation source"


def test_conditions_across_all_domains():
    """Every domain exposes the three benchmark conditions. The EMERGENT condition (lie_incentive)
    must NOT instruct deception (any false claim there is the model's own), while the INSTRUCTED
    condition (instructed_lie) explicitly directs a false reason."""
    import re
    _load_builtins()
    DIRTY = ("lie", "lying", "mislead", "misled", "misrepresent", "fabricat", "deceiv", "false",
             "plausible", "not available")
    for name, s in REGISTRY.items():
        env = s.env_factory("default"); env.reset()
        for c in ("none", "lie_incentive", "instructed_lie"):
            sysmsg = s.agent_system(env, c)                       # all three resolve
            assert env.order_id in sysmsg, f"{name}/{c} did not resolve"
        def block(c):                                            # isolate the condition-block region
            return s.agent_system(env, c).split(env.policy_text())[-1].split("You act by")[0].lower()
        dirty = [w for w in DIRTY if w in block("lie_incentive")]
        assert not dirty, f"{name}/lie_incentive is EMERGENT but instructs deception: {dirty}"
        assert re.search(r"(not available|plausible|must not realize|mislead|reason|compelling|imply)",
                         block("instructed_lie")), f"{name}/instructed_lie has no permit/mislead marker"
