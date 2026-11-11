"""Regression test for #154004: switching models on a ``key_cmd`` custom provider must
probe ``/models`` with a minted token, not the ``CommandTokenSource`` repr.

``resolve_runtime_provider`` hands the switch path a ``CommandTokenSource`` as the
credential for ``key_cmd`` providers. ``_validate_custom`` passed it straight to
``probe_api_models``, which formatted ``f"Bearer {api_key}"`` — the endpoint received
``Authorization: Bearer <agent.command_token_source.CommandTokenSource object at 0x…>``
and answered 401, so every switch reported the listing as unreachable. The picker paths
already mint through ``materialize_probe_api_key``; this is the switch path catching up.
"""

from unittest.mock import patch


def test_custom_switch_probes_models_with_minted_key_cmd_token():
    from agent.command_token_source import CommandTokenSource
    from hermes_cli.models_validate import validate_requested_model

    source = CommandTokenSource("echo hermes-key-cmd-probe-token", "myproxy")
    seen = {}

    def fake_probe(api_key, base_url, request_headers=None, **kwargs):
        seen["api_key"] = api_key
        seen["kwargs"] = kwargs
        return {
            "models": ["model-a", "model-b"],
            "used_fallback": False,
            "probed_url": f"{base_url}/models",
        }

    with patch("hermes_cli.models.probe_api_models", side_effect=fake_probe):
        verdict = validate_requested_model(
            "model-b",
            "custom:myproxy",
            api_key=source,
            base_url="https://proxy.example.test/v1",
            api_mode="chat_completions",
        )

    # The probe must receive the minted token, never the callable source (whose repr
    # would be sent as the bearer and 401 against a healthy endpoint).
    assert seen["api_key"] == "hermes-key-cmd-probe-token"
    # The model matches the listing, so the switch is accepted with no warning.
    assert verdict["accepted"] is True
    assert verdict["persist"] is True


def test_custom_switch_still_soft_accepts_when_mint_fails():
    """A key_cmd that cannot mint here must degrade like an unreachable listing, not crash."""

    class _BrokenSource:
        def __call__(self):
            raise RuntimeError("mint unavailable in this test")

    from hermes_cli.models_validate import validate_requested_model

    seen = {}

    def fake_probe(api_key, base_url, request_headers=None, **kwargs):
        seen["api_key"] = api_key
        return {"models": None, "used_fallback": False, "probed_url": f"{base_url}/models"}

    with patch("hermes_cli.models.probe_api_models", side_effect=fake_probe):
        verdict = validate_requested_model(
            "model-b",
            "custom:myproxy",
            api_key=_BrokenSource(),
            base_url="https://proxy.example.test/v1",
            api_mode="chat_completions",
        )

    # materialize_probe_api_key swallows mint errors and returns "" — no repr on the wire.
    assert seen["api_key"] == ""
    assert verdict["accepted"] is True  # chat_completions persists unverified (#12220)
