from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_environment() -> None:
    """Load local environment variables from .env when the file exists."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


def _read_env(name: str, default: str) -> str:
    """Read an environment variable and fall back to a safe default."""
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """Central application settings for MarketLens."""

    environment: str
    provider: str
    default_period: str
    default_interval: str
    cache_dir: Path
    log_level: str


def load_settings() -> Settings:
    """Load and validate project settings from environment variables."""
    _load_environment()

    environment = _read_env("MARKETLENS_ENV", "development")
    provider = _read_env("MARKETLENS_PROVIDER", "yfinance")
    default_period = _read_env("MARKETLENS_DEFAULT_PERIOD", "5y")
    default_interval = _read_env("MARKETLENS_DEFAULT_INTERVAL", "1d")
    cache_dir_value = _read_env("MARKETLENS_CACHE_DIR", "data/cache")
    log_level = _read_env("MARKETLENS_LOG_LEVEL", "INFO").upper()

    if provider != "yfinance":
        raise ValueError(
            "MARKETLENS_PROVIDER must currently be 'yfinance'. "
            f"Received: {provider!r}"
        )

    if default_interval not in {"1d", "1wk", "1mo"}:
        raise ValueError(
            "MARKETLENS_DEFAULT_INTERVAL must be one of: "
            "'1d', '1wk', or '1mo'. "
            f"Received: {default_interval!r}"
        )

    cache_dir = PROJECT_ROOT / cache_dir_value
    cache_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        environment=environment,
        provider=provider,
        default_period=default_period,
        default_interval=default_interval,
        cache_dir=cache_dir,
        log_level=log_level,
    )