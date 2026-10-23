"""FastAPI 앱 조립: 시작할 때 저장소에 접속하고 검색 서비스를 만든다.

실행:
  uv run uvicorn playground_wiki.api.app:create_app --factory --host 127.0.0.1 --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from playground_retrieval.search import SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import ApiConfig
from playground_wiki.api.routes import search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """모델 · 색인을 올려 app.state 에 두고, 끝날 때 PG 접속을 닫는다."""
    config = ApiConfig.from_env()
    logger.info("loading search service for collection %s", config.search.collection)
    with PgStore.connect(config.pg) as pg:
        app.state.search_service = SearchService.build(config.search, pg, QdrantStore.from_env())
        yield


def create_app() -> FastAPI:
    """라우터를 붙인 앱. lifespan 은 서버가 시작할 때 돈다 (TestClient 를 with 없이 쓰면 돌지 않는다)."""
    app = FastAPI(title="llm-playground-wiki", lifespan=lifespan)
    app.include_router(search.router)
    return app
