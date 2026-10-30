import base64
import importlib
from dataclasses import FrozenInstanceError

import pytest

from cv_agent import settings as settings_module
from cv_agent.settings import Settings, settings


def test_settings_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(settings, "model", "other")


def test_settings_defaults() -> None:
    s = Settings(model="m", base_url="http://x")
    assert (s.model, s.base_url) == ("m", "http://x")
    assert s.max_pages == 3
    assert s.min_text_chars < s.max_text_chars


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://remote:1234")
    try:
        reloaded = importlib.reload(settings_module)
        assert reloaded.settings.model == "custom-model"
        assert reloaded.settings.base_url == "http://remote:1234"
    finally:
        monkeypatch.undo()
        importlib.reload(settings_module)


def test_base64_roundtrip_helper(pdf_bytes: bytes) -> None:
    assert base64.b64decode(base64.b64encode(pdf_bytes)) == pdf_bytes
