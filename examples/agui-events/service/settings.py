"""Everything this service reads from the environment, in one place."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration, from the environment or a ``.env``.

    ``provider_model`` and ``provider_api_key`` are the same two names the CLI
    and the Chainlit host read. Both are **optional here**, unlike a real
    deployment, because the point of the example is that it runs with neither:
    without them a scripted stub answers by keyword. See :mod:`service.model`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_model: str | None = None
    provider_api_key: SecretStr | None = None

    port: int = Field(default=8765, ge=1, le=65535)

    #: Empty because the browser talks to Vite, which proxies `/api` here — so
    #: it is one origin and there is nothing to allow. A client served from
    #: anywhere else sets this.
    allowed_origins: str = ""

    #: Pause between the stub's tokens. A stub has none and a real provider has
    #: tens of milliseconds, so without this an answer arrives faster than a
    #: screen can draw it, and streaming that works looks exactly like
    #: streaming that does not. Ignored when a real model runs.
    token_delay: float = 0.04


@lru_cache
def get_settings() -> Settings:
    """The settings, read once per process."""
    return Settings()
