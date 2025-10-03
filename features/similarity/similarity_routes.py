import logging
from fastapi import APIRouter, HTTPException, Query
from features.similarity.similarity_service import find_similar_notes


router = APIRouter()


@router.get("/similarity/notes/{note_id}")
async def find_similar_notes_route(note_id: int, limit: int = 10, threshold: float = 0.65):
    try:
        results = find_similar_notes(note_id=note_id, limit=limit, threshold=threshold)
        return {"results": results}
    except Exception as e:
        logging.error(f"failed to find similar notes for note {note_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
