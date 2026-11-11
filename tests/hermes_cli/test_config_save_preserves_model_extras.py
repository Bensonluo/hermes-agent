"""A whole-file save must not drop unknown keys nested in a dict-valued ``model.default``.

Issue #153296: ``_normalize_root_model_keys`` flattens a dict-valued ``model.default`` to its
``model``/``default`` string plus ``provider`` and drops every other nested key, so any
whole-file save (``save_config(load_config())``, ``hermes tools disable web``, ...) silently
loses forward-compatible fields a newer Hermes wrote. The unknown nested keys are lifted into
the ``model`` section next to ``default`` instead of being discarded.
"""

import pytest
import hermes_yaml as yaml

import hermes_cli.config as config_mod
from hermes_cli.config import DEFAULT_CONFIG, load_config, save_config

_STAMP = DEFAULT_CONFIG["_config_version"]

_CONFIG = f"""\
_config_version: {_STAMP}
model:
  default: {{model: x/y, provider: openrouter, newer_field: keepme}}
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    (tmp_path / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    yield tmp_path
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def _write(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")


def _read(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


def test_save_preserves_unknown_keys_nested_in_dict_valued_model_default(home):
    save_config(load_config())

    saved = _read(home)
    assert saved["model"]["default"] == "x/y"
    assert saved["model"]["provider"] == "openrouter"
    assert saved["model"]["newer_field"] == "keepme"


def test_save_preserves_unknown_keys_nested_in_model_name_alias(home):
    _write(home, f"_config_version: {_STAMP}\nmodel:\n  name: {{model: a/b, provider: zai, extra_key: 7}}\n")

    save_config(load_config())

    saved = _read(home)
    assert saved["model"]["default"] == "a/b"
    assert saved["model"]["provider"] == "zai"
    assert saved["model"]["extra_key"] == 7


def test_second_save_is_stable_and_keeps_lifted_keys(home):
    save_config(load_config())
    save_config(load_config())

    assert _read(home)["model"]["newer_field"] == "keepme"


def test_nested_consumed_keys_do_not_override_explicit_model_section_keys(home):
    _write(home, f"_config_version: {_STAMP}\n"
                 "model:\n"
                 "  provider: anthropic\n"
                 f"  default: {{model: x/y, provider: openrouter, newer_field: keepme}}\n")

    save_config(load_config())

    saved = _read(home)
    # An explicit configured provider wins over the nested one (existing precedence).
    assert saved["model"]["provider"] == "anthropic"
    assert saved["model"]["default"] == "x/y"
    assert saved["model"]["newer_field"] == "keepme"


def test_nested_id_alias_keys_keep_flatten_only_behavior(home):
    # ``model``/``default``/``name``/``provider`` inside the nested dict are consumed by the
    # flatten itself; they are not lifted, so a degenerate nested-alias input behaves exactly
    # as before (no id is invented where none was configured).
    _write(home, f"_config_version: {_STAMP}\nmodel:\n  default: {{name: ghost}}\n")

    save_config(load_config())

    assert _read(home)["model"].get("default", "") == ""
