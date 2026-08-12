"""``python -m service`` — the same app ``uvicorn service.app:app`` serves."""

import uvicorn

from service.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "service.app:app",
        host="127.0.0.1",
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
