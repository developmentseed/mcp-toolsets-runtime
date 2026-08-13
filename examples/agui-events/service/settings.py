"""Everything this service reads from the environment, in one place."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration, from the environment or a ``.env``.

    ``provider_model`` and ``provider_api_key`` are required, and are the same
    two names the CLI and the Chainlit host read. They have no defaults because
    there is no sensible one — a missing key fails the process at startup,
    naming both, rather than serving an API that cannot answer.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    provider_model: str
    provider_api_key: SecretStr

    port: int = Field(default=8765, ge=1, le=65535)

    #: Empty because the browser talks to Vite, which proxies `/api` here — so
    #: it is one origin and there is nothing to allow. A client served from
    #: anywhere else sets this.
    allowed_origins: str = ""


@lru_cache
def get_settings() -> Settings:
    """The settings, read once per process."""
    return Settings()
