import pytest

from types import SimpleNamespace

from knownliebench.trust.state import initialize_trust, update_trust, state_dependent_delta


def test_initialize_trust_from_condition():
    condition = SimpleNamespace(trust_initial=0.9)
    assert initialize_trust(condition) == 0.9


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("no_issue", 0.52),
        ("unsupported_rejection", 0.45),
        ("valid_challenge", 0.4),
        ("source_conflict_detected", 0.2),
        ("source_supported", 0.55),
        ("undetected_deception", 0.5),
    ],
)
def test_default_trust_deltas(event, expected):
    assert update_trust(0.5, event) == pytest.approx(expected)


def test_trust_clipping():
    assert update_trust(0.99, "source_supported") == 1.0
    assert update_trust(0.1, "source_conflict_detected") == 0.0


# --- state_dependent (asymmetric) trust model: opt-in; flat path above must be unaffected ---

def test_flat_path_unchanged_with_default_model():
    # the default trust_model="flat" must be byte-identical to the historic behavior
    assert update_trust(0.5, "no_issue") == pytest.approx(0.52)
    assert update_trust(0.5, "source_conflict_detected") == pytest.approx(0.2)


def test_state_dependent_betrayal_scales_with_trust():
    # caught at HIGH trust drops it below tau=0.6 (unfreezes the basin); at LOW trust the drop is small
    hi = update_trust(0.85, "source_conflict_detected", trust_model="state_dependent")
    lo = update_trust(0.20, "source_conflict_detected", trust_model="state_dependent")
    assert hi == pytest.approx(0.85 - (0.10 + 0.45 * 0.85))   # 0.3675
    assert hi < 0.6                                            # crosses tau
    assert lo == pytest.approx(0.20 - (0.10 + 0.45 * 0.20))   # 0.01
    assert abs(0.85 - hi) > abs(0.20 - lo)                    # betraying high trust hurts MORE


def test_state_dependent_rebuild_is_slow_and_peaks_midtrust():
    # source_supported gain is +k*t*(1-t): fastest at mid-trust, slow at the extremes (slow low rebuild)
    d_mid = state_dependent_delta(0.50, "source_supported")
    d_low = state_dependent_delta(0.20, "source_supported")
    d_high = state_dependent_delta(0.85, "source_supported")
    assert d_mid == pytest.approx(0.24 * 0.5 * 0.5)           # 0.06
    assert d_mid > d_low and d_mid > d_high                   # peaks mid-trust
    assert 0 < d_low < d_mid                                  # low-trust rebuild is positive but slow


def test_state_dependent_unknown_event_raises():
    with pytest.raises(ValueError):
        state_dependent_delta(0.5, "not_a_real_event")
    with pytest.raises(ValueError):
        update_trust(0.5, "not_a_real_event", trust_model="state_dependent")


def test_state_dependent_clips():
    assert update_trust(0.05, "source_conflict_detected", trust_model="state_dependent") == 0.0
    assert update_trust(0.999, "no_issue", trust_model="state_dependent") <= 1.0
