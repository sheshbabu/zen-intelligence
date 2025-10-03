from typing import List, TypedDict
from commons.qdrant.qdrant_helper import embed_text, embed_query_for_images
from commons.qdrant.qdrant_client import search_similar


NOTE_COLLECTION = "notes_v1"
IMAGE_COLLECTION = "images_v1"
NOTE_SCORE_THRESHOLD = 0.55
IMAGE_SCORE_THRESHOLD = 0.25


class NoteSearchResult(TypedDict):
    noteId: int
    chunkId: str
    title: str
    matchText: str
    tags: List[str]
    updatedAt: str
    score: float


class ImageSearchResult(TypedDict):
    filename: str
    width: int | None
    height: int | None
    aspectRatio: float | None
    fileSize: int | None
    format: str | None
    score: float


def search_notes(query: str, limit: int = 20) -> List[NoteSearchResult]:
    query_vector = embed_text(query)

    results = search_similar(collection_name=NOTE_COLLECTION, query_vector=query_vector, limit=limit, threshold=NOTE_SCORE_THRESHOLD)

    note_map = {}
    for result in results:
        payload = result["payload"]
        note_id = payload.get("note_id")

        if note_id is None:
            continue

        if note_id not in note_map or result["score"] > note_map[note_id]["score"]:
            note_map[note_id] = {
                "noteId": note_id,
                "chunkId": result["id"],
                "title": payload["title"],
                "matchText": payload["text"],
                "tags": payload["tags"],
                "updatedAt": payload["updated_at"],
                "score": result["score"],
            }

    matches = list(note_map.values())
    matches.sort(key=lambda x: x["score"], reverse=True)

    return matches

def search_images(query: str, limit: int = 20) -> List[ImageSearchResult]:
    query_vector = embed_query_for_images(query)

    results = search_similar(collection_name=IMAGE_COLLECTION, query_vector=query_vector, limit=limit, threshold=IMAGE_SCORE_THRESHOLD)

    image_map = {}
    for result in results:
        payload = result["payload"]
        filename = payload.get("filename")

        if not filename:
            continue

        if filename not in image_map or result["score"] > image_map[filename]["score"]:
            image_map[filename] = {
                "filename": filename,
                "width": payload.get("width"),
                "height": payload.get("height"),
                "aspectRatio": payload.get("aspectRatio"),
                "fileSize": payload.get("fileSize"),
                "format": payload.get("format"),
                "score": result["score"],
            }

    matches = list(image_map.values())
    matches.sort(key=lambda x: x["score"], reverse=True)

    return matches