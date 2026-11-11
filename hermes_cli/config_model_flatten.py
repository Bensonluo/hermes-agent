"""Flatten dict-valued ``model.default``/``model.model``/``model.name`` for readers.

A newer Hermes may persist ``model.default`` as a mapping (``{model: ..., provider: ..., <new
fields>}``). Readers only understand the scalar form, so the load/save chokepoint canonicalizes
the value to its model string and provider. Keys this build does not know are *lifted* into the
``model`` section as siblings of ``default`` instead of being silently discarded, so a
whole-file save by an older build does not strip forward-compatible fields a newer build wrote
(issue #153296). Lifted keys survive subsequent loads and saves unchanged.
"""

from __future__ import annotations

from typing import Any

#: Nested keys the flatten itself consumes (``model``/``default``/``name`` are id aliases,
#: ``provider`` the routing target); anything else is forward-compatible data this build
#: cannot interpret and must preserve.
_CONSUMED_NESTED_KEYS = frozenset({"model", "default", "provider", "name"})


def flatten_model_default_dicts(model: dict[str, Any]) -> None:
    """Canonicalize dict-valued ``default``/``model``/``name`` entries of *model* in place.

    The nested provider wins over a merged default ``"auto"`` (which runtime resolution treats
    as authoritative) but never over a configured one. Unknown nested keys are lifted next to
    ``default`` (never overriding an explicit ``model.*`` key) so a whole-file save keeps them.
    """
    for key in ("default", "model", "name"):
        value = model.get(key)
        if not isinstance(value, dict):
            continue
        for nested_key, nested_value in value.items():
            if nested_key not in _CONSUMED_NESTED_KEYS and nested_key not in model:
                model[nested_key] = nested_value
        nested_model = value.get("model") or value.get("default")
        nested_provider = str(value.get("provider") or "").strip()
        model[key] = str(nested_model or "").strip()
        if nested_provider:
            outer_provider = str(model.get("provider") or "").strip()
            if not outer_provider or outer_provider == "auto":
                model["provider"] = nested_provider
