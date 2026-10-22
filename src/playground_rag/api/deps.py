"""라우터가 Depends 로 받는 의존성. lifespan 이 app.state 에 둔 객체를 꺼내 준다."""

from typing import Annotated

from fastapi import Depends, Request

from playground_rag.retrieval import SearchService


def get_search_service(request: Request) -> SearchService:
    """앱이 시작할 때 lifespan 이 만든 검색 서비스."""
    return request.app.state.search_service


SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
