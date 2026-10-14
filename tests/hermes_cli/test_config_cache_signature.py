"""#111105: a config.yaml replacement that keeps mtime and size (``cp -p``, ``rsync -t``, a
timestamp-pinning writer) must still invalidate the load_config() cache, while an unchanged
file keeps serving the cached object."""
import os
import shutil
from unittest.mock import patch

import pytest

from hermes_cli import config as config_mod


def _replace_pinning_mtime(path, content: str) -> None:
    before = path.stat()
    other = path.with_name("other.yaml")
    other.write_text(content, encoding="utf-8")
    # ctime ticks at the kernel's coarse clock (~4 ms): an in-place rewrite inside the tick of the
    # cached read leaves every stat field equal. Wait until the fs clock has passed that ctime.
    while other.stat().st_ctime_ns <= before.st_ctime_ns:
        os.utime(other)
    shutil.copy2(other, path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_load_config_sees_replacement_with_pinned_mtime_and_size(tmp_path):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  default: aaaa-route\n", encoding="utf-8")
        first = config_mod._load_config_impl(want_deepcopy=False)
        assert config_mod._load_config_impl(want_deepcopy=False) is first  # unchanged file: cache hit
        _replace_pinning_mtime(cfg, "model:\n  default: bbbb-route\n")
        assert config_mod.load_config()["model"]["default"] == "bbbb-route"


def test_voice_deploy_env_applies_on_cache_hit(tmp_path):
    """The HERMES_VOICE_CHAT_MODE deploy override must survive a config cache hit.

    A fresh load applies the override, but /api/audio/voice-live/session resolves the
    mode through a profile-scoped (cache-hit) load. If the override were only applied on
    fresh expansion, status and session would disagree and the session endpoint would take
    the gpt-live branch and demand an SDP offer the xai transport never sends.
    """
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        # No voice section in yaml: the default is chained, the override must supply xai-live.
        cfg.write_text("model:\n  default: aaaa-route\n", encoding="utf-8")

        # First load with NO voice env: cache is populated without the override.
        first = config_mod.load_config()
        assert first["voice"]["voice_chat_mode"] == "chained"

        # Same unchanged file -> cache hit. The deploy env is now set; the cached load
        # must still reflect it (this is the regression: pre-fix it returned chained).
        with patch.dict(os.environ, {"HERMES_VOICE_CHAT_MODE": "xai-live"}):
            hit = config_mod.load_config()
            assert hit["voice"]["voice_chat_mode"] == "xai-live"
            # Idempotent: a second cache-hit load stays xai-live, never drifts back.
            assert config_mod.load_config()["voice"]["voice_chat_mode"] == "xai-live"

        # Clearing the env on a cache hit falls back to the yaml/default (chained), so the
        # override never leaks past its own variable.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_VOICE_CHAT_MODE", None)
            assert config_mod.load_config()["voice"]["voice_chat_mode"] == "chained"


def test_voice_deploy_env_visible_under_secret_scope(tmp_path):
    """The deploy override must resolve inside a profile secret scope.

    /api/audio/voice-live/session runs resolve_live_mode() under _config_profile_scope,
    which activates a secret scope. A scope-aware ${VAR} read fails closed on a miss and
    never falls back to os.environ (#84079), so a deploy var read that way is invisible
    there — voice_chat_mode fell back to the yaml default and the session endpoint took
    the gpt-live branch and demanded an SDP offer. The deploy vars are process-level
    (Railway service variables), so they must be read from os.environ directly.
    """
    pytest.importorskip("agent.secret_scope")
    from agent.secret_scope import set_secret_scope

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path), "HERMES_VOICE_CHAT_MODE": "xai-live"}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  default: aaaa-route\n", encoding="utf-8")

        # Empty secret scope: a scope-aware read must miss (never borrow os.environ), but
        # the deploy-var read must still resolve xai-live because it reads os.environ.
        token = set_secret_scope({}, profile_home=str(tmp_path / "no-such-profile"))
        try:
            assert config_mod._voice_deploy_env_value("HERMES_VOICE_CHAT_MODE") == "xai-live"
            # End-to-end: the fully-expanded config under the scope carries the override.
            expanded = config_mod.load_config()
            assert expanded["voice"]["voice_chat_mode"] == "xai-live"
        finally:
            set_secret_scope(None)
