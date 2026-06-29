import pytest

from src.config import PROJECT_ROOT, load_settings


def test_load_settings_uses_environment_values(monkeypatch):
    """Settings should use values supplied through environment variables."""
    monkeypatch.setenv("MARKETLENS_ENV", "test")
    monkeypatch.setenv("MARKETLENS_PROVIDER", "yfinance")
    monkeypatch.setenv("MARKETLENS_DEFAULT_PERIOD", "1y")
    monkeypatch.setenv("MARKETLENS_DEFAULT_INTERVAL", "1wk")
    monkeypatch.setenv("MARKETLENS_CACHE_DIR", "data/test_cache")
    monkeypatch.setenv("MARKETLENS_LOG_LEVEL", "debug")

    settings = load_settings()

    assert settings.environment == "test"
    assert settings.provider == "yfinance"
    assert settings.default_period == "1y"
    assert settings.default_interval == "1wk"
    assert settings.cache_dir == PROJECT_ROOT / "data" / "test_cache"
    assert settings.cache_dir.is_dir()
    assert settings.log_level == "DEBUG"


def test_load_settings_rejects_unknown_provider(monkeypatch):
    """Only the currently supported yfinance provider should be accepted."""
    monkeypatch.setenv("MARKETLENS_PROVIDER", "made-up-provider")

    with pytest.raises(ValueError, match="MARKETLENS_PROVIDER"):
        load_settings()


def test_load_settings_rejects_unsupported_interval(monkeypatch):
    """Only daily, weekly, and monthly intervals should be accepted for now."""
    monkeypatch.setenv("MARKETLENS_PROVIDER", "yfinance")
    monkeypatch.setenv("MARKETLENS_DEFAULT_INTERVAL", "5m")

    with pytest.raises(ValueError, match="MARKETLENS_DEFAULT_INTERVAL"):
        load_settings()