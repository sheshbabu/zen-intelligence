import logging
from typing import List, Dict, Any
from collections import defaultdict
from commons.qdrant.qdrant_client import scroll_points, search_similar_with_filter
from commons.qdrant.qdrant_helper import NOTE_COLLECTION


def find_similar_notes(note_id: int, limit: int = 10, threshold: float = 0.5) -> List[Dict[str, Any]]:
    source_chunks = scroll_points(NOTE_COLLECTION, {"note_id": note_id}, limit=1000, with_vectors=True)

    if not source_chunks:
        logging.warning(f"No chunks found for note {note_id}")
        return []

    logging.info(f"Finding similar notes for note {note_id} with {len(source_chunks)} chunks")

    note_scores = defaultdict(lambda: {"max_score": 0.0, "match_count": 0, "payload": None})

    for i, chunk in enumerate(source_chunks):
        vector = chunk["vector"]

        if not vector:
            logging.warning(f"Chunk {i} has no vector")
            continue

        if not isinstance(vector, list):
            vector = list(vector)

        logging.debug(f"Searching with chunk {i} (vector len={len(vector)}): {chunk['payload'].get('text', '')[:80]}...")

        similar_chunks = search_similar_with_filter(collection_name=NOTE_COLLECTION, query_vector=vector, filter={}, limit=limit * 3, threshold=threshold)

        if similar_chunks:
            top_match = similar_chunks[0]
            logging.info(f"Chunk {i} found {len(similar_chunks)} similar chunks. Top: note_id={top_match['payload'].get('note_id')}, score={top_match['score']:.4f}")
        else:
            logging.info(f"Chunk {i} found 0 similar chunks")

        for result in similar_chunks:
            result_note_id = result["payload"].get("note_id")

            if result_note_id == note_id:
                continue

            score = result["score"]
            note_scores[result_note_id]["match_count"] += 1

            if score > note_scores[result_note_id]["max_score"]:
                note_scores[result_note_id]["max_score"] = score
                note_scores[result_note_id]["payload"] = result["payload"]

    results = [
        {
            "note_id": note_id,
            "max_score": data["max_score"],
            "match_count": data["match_count"],
            "title": data["payload"].get("title", ""),
            "tags": data["payload"].get("tags", []),
            "updated_at": data["payload"].get("updated_at", ""),
        }
        for note_id, data in note_scores.items()
        if data["payload"] is not None
    ]

    results.sort(key=lambda x: (x["max_score"], x["match_count"]), reverse=True)

    results = results[:limit]

    logging.info(f"Found {len(results)} similar notes for note {note_id}")
    return results
