import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from features.search.search_service import search_notes, search_images


router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


@router.post("/search/notes")
async def search_notes_route(request: SearchRequest):
    try:
        results = search_notes(query=request.query, limit=request.limit)
        return {"results": results}
    except Exception as e:
        logging.error(f"failed to search notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/images")
async def search_images_route(request: SearchRequest):
    try:
        results = search_images(query=request.query, limit=request.limit)
        return {"results": results}
    except Exception as e:
        logging.error(f"failed to search images: {e}")
        raise HTTPException(status_code=500, detail=str(e))