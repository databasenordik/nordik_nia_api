import asyncio
import os
import logging

import uvicorn

from app.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    port = int(os.environ.get("PORT", settings.app_port))
    if settings.nia_transport == "grpc":
        from app.grpc_api.server import serve

        asyncio.run(serve(settings, port))
    else:
        uvicorn.run("app.main:app", host=settings.app_host, port=port)


if __name__ == "__main__":
    main()
