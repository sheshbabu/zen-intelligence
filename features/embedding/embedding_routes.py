import logging
import time
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from features.embedding.embedding_service import (process_note, process_image, delete_note_embeddings, delete_image_embeddings)
router = APIRouter()


class EmbedNoteRequest(BaseModel):
    title: str
    content: str
    tags: List[str]
    updated_at: str


class EmbedImageRequest(BaseModel):
    filename: str
    image_path: str
    width: int
    height: int
    aspect_ratio: float
    file_size: int
    format: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 20


@router.post("/embed/notes/{note_id}")
async def embed_note_route(note_id: int, request: EmbedNoteRequest):
    try:
        start = time.time()
        process_note(note_id=note_id, title=request.title, content=request.content, tags=request.tags, updated_at=request.updated_at)
        elapsed = time.time() - start
        logging.info(f"embedded note: {request.title} ({elapsed:.2f}s)")
        return {"success": True}
    except Exception as e:
        logging.error(f"failed to embed note {note_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed/images/{filename}")
async def embed_image_route(filename: str, request: EmbedImageRequest):
    try:
        start = time.time()
        process_image(
            filename=filename,
            image_path=request.image_path,
            width=request.width,
            height=request.height,
            aspect_ratio=request.aspect_ratio,
            file_size=request.file_size,
            format=request.format
        )
        elapsed = time.time() - start
        logging.info(f"embedded image: {filename} ({elapsed:.2f}s)")
        return {"success": True}
    except Exception as e:
        logging.error(f"failed to embed image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/embed/notes/{note_id}")
async def delete_note_route(note_id: int):
    try:
        delete_note_embeddings(note_id)
        return {"success": True}
    except Exception as e:
        logging.error(f"failed to delete note embeddings {note_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/embed/images/{filename}")
async def delete_image_route(filename: str):
    try:
        delete_image_embeddings(filename)
        return {"success": True}
    except Exception as e:
        logging.error(f"failed to delete image embeddings {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))