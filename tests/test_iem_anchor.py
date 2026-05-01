from __future__ import annotations

from civitasos_runtime.iem_anchor import build_iem_anchor, genesis_iem_state, iem_state_hash


def test_iem_state_hash_is_stable_for_key_order() -> None:
    left = {"b": 2, "a": {"y": 1, "x": 0}}
    right = {"a": {"x": 0, "y": 1}, "b": 2}

    assert iem_state_hash(left) == iem_state_hash(right)


def test_build_iem_anchor_is_replayable_from_state_and_log() -> None:
    state = genesis_iem_state("did:civ:testnet:agent")
    update_log = [{"target": "relation", "new_value": 0.7}]

    first = build_iem_anchor(
        identity_id="did:civ:testnet:agent",
        state=state,
        update_log=update_log,
    )
    second = build_iem_anchor(
        identity_id="did:civ:testnet:agent",
        state=dict(reversed(list(state.items()))),
        update_log=list(update_log),
    )

    assert first == second
    assert first.version_id.startswith("iem:v1:")
    assert first.state_hash.startswith("sha256:")
    assert first.latest_update_log_hash.startswith("sha256:")
    assert first.storage_hint == "civitasos://identity/did:civ:testnet:agent/iem/latest"