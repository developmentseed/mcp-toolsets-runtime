"""The chat model behind the API.

`init_chat_model` is provider-agnostic, so `PROVIDER_MODEL` decides which
provider is used and the runtime declares none of their SDKs — a deployment
installs the driver it uses. `mistral-small-latest` needs `langchain-mistralai`
on the path, `openai:gpt-4o-mini` needs `langchain-openai`, and so on.
"""

from typing import Any

from langchain.chat_models import init_chat_model

from service.settings import get_settings


def build() -> tuple[Any, str]:
    """The model to run with, and its name for the startup log."""
    settings = get_settings()
    return (
        init_chat_model(
            settings.provider_model,
            api_key=settings.provider_api_key.get_secret_value(),
        ),
        settings.provider_model,
    )
