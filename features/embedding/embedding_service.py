import logging
import uuid
from typing import List
from features.chunking.chunking_service import chunk_note
from commons.qdrant.qdrant_client import (upsert_points, delete_points_by_filter)
from commons.qdrant.qdrant_helper import NOTE_COLLECTION, IMAGE_COLLECTION, embed_image, embed_text

def process_note(note_id: int, title: str, content: str, tags: List[str], updated_at: str) -> None:
    delete_note_embeddings(note_id)

    chunks = chunk_note(content)
    if not chunks:
        return

    points = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        point = {
            "id": str(uuid.uuid4()),
            "vector": embedding,
            "payload": {
                "text": chunk,
                "note_id": note_id,
                "chunk_index": i,
                "title": title,
                "tags": tags,
                "updated_at": updated_at,
            }
        }
        points.append(point)

    upsert_points(NOTE_COLLECTION, points)
    logging.debug(f"Processed note {note_id} with {len(chunks)} chunks")


def process_image(filename: str, image_path: str, width: int, height: int, aspect_ratio: float, file_size: int, format: str) -> None:
    delete_image_embeddings(filename)

    embedding = embed_image(image_path)

    payload = {
        "filename": filename,
        "width": width,
        "height": height,
        "aspectRatio": aspect_ratio,
        "fileSize": file_size,
        "format": format,
    }

    point = {
        "id": str(uuid.uuid4()),
        "vector": embedding,
        "payload": payload
    }

    upsert_points(IMAGE_COLLECTION, [point])
    logging.debug(f"Processed image {filename}")


def delete_note_embeddings(note_id: int) -> None:
    delete_points_by_filter(NOTE_COLLECTION, {"note_id": note_id})
    logging.debug(f"Deleted embeddings for note {note_id}")


def delete_image_embeddings(filename: str) -> None:
    delete_points_by_filter(IMAGE_COLLECTION, {"filename": filename})
    logging.debug(f"Deleted embeddings for image {filename}")