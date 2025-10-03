import logging
import numpy as np
from typing import List, TypedDict, Dict, Any
from sklearn.cluster import DBSCAN
from commons.qdrant.qdrant_client import scroll_points, search_similar
from commons.qdrant.qdrant_helper import NOTE_COLLECTION


OUTLIER_SCORE_THRESHOLD = 0.5


class NoteScore(TypedDict):
    max_score: float
    outlier_matches: int
    routine_matches: int
    payload: Dict[str, Any] | None


class SimilarNoteResult(TypedDict):
    note_id: int
    max_score: float
    outlier_matches: int
    routine_matches: int
    weighted_score: int
    title: str
    tags: List[str]
    updated_at: str


class OutlierChunk(TypedDict):
    chunk_id: str
    text: str
    chunk_index: int
    outlier_score: float
    cluster_label: int


def find_similar_notes(note_id: int, limit: int = 10, threshold: float = 0.5) -> List[SimilarNoteResult]:
    source_chunks = scroll_points(NOTE_COLLECTION, {"note_id": note_id}, limit=1000, with_vectors=True)

    if not source_chunks:
        return []

    logging.debug(f"Finding similar notes for note {note_id} with {len(source_chunks)} chunks")

    outlier_chunks = find_outlier_chunks(source_chunks)
    outlier_chunk_ids = {c['chunk_id'] for c in outlier_chunks if c['outlier_score'] >= OUTLIER_SCORE_THRESHOLD}

    logging.debug(f"Found {len(outlier_chunk_ids)} outlier chunks (score >= {OUTLIER_SCORE_THRESHOLD})")

    note_scores: Dict[int, NoteScore] = {}

    for i, chunk in enumerate(source_chunks):
        vector = chunk["vector"]

        if not vector:
            logging.warning(f"Chunk {i} has no vector")
            continue

        is_outlier = chunk["id"] in outlier_chunk_ids
        chunk_type = "outlier" if is_outlier else "routine"

        logging.debug(f"Searching with {chunk_type} chunk {i}: {chunk['payload'].get('text', '')[:80]}...")

        similar_chunks = search_similar(
            collection_name=NOTE_COLLECTION,
            query_vector=vector,
            limit=limit * 3,
            threshold=threshold
        )

        if similar_chunks:
            top_match = next((match for match in similar_chunks if match['payload'].get('note_id') != note_id), None)

            if top_match:
                logging.debug(f"{chunk_type.capitalize()} chunk {i} found {len(similar_chunks)} matches. Top: note_id={top_match['payload'].get('note_id')}, score={top_match['score']:.4f}")
            else:
                logging.debug(f"{chunk_type.capitalize()} chunk {i} found {len(similar_chunks)} matches (all from same note)")
        else:
            logging.debug(f"{chunk_type.capitalize()} chunk {i} found 0 matches")

        for result in similar_chunks:
            result_note_id = result["payload"].get("note_id")

            if result_note_id == note_id:
                continue

            score = result["score"]

            if result_note_id not in note_scores:
                note_scores[result_note_id] = {
                    "max_score": 0.0,
                    "outlier_matches": 0,
                    "routine_matches": 0,
                    "payload": None
                }

            if is_outlier:
                note_scores[result_note_id]["outlier_matches"] += 1
            else:
                note_scores[result_note_id]["routine_matches"] += 1

            if score > note_scores[result_note_id]["max_score"]:
                note_scores[result_note_id]["max_score"] = score
                note_scores[result_note_id]["payload"] = result["payload"]

    results = []
    for note_id, data in note_scores.items():
        if data["payload"] is None:
            continue

        weighted_score = (data["outlier_matches"] * 3) + data["routine_matches"]

        results.append({
            "note_id": note_id,
            "max_score": data["max_score"],
            "outlier_matches": data["outlier_matches"],
            "routine_matches": data["routine_matches"],
            "weighted_score": weighted_score,
            "title": data["payload"].get("title", ""),
            "tags": data["payload"].get("tags", []),
            "updated_at": data["payload"].get("updated_at", ""),
        })

    results.sort(key=lambda x: (x["weighted_score"], x["max_score"]), reverse=True)

    results = results[:limit]

    logging.info(f"Found {len(results)} similar notes for note {note_id}")
    return results


def find_outlier_chunks(chunks: List[Dict[str, Any]], eps: float = 0.3, min_samples: int = 3) -> List[OutlierChunk]:
    if not chunks:
        logging.warning(f"No chunks provided")
        return []

    if len(chunks) < 5:
        logging.info(f"Only {len(chunks)} chunks - too few for clustering")
        return []

    logging.info(f"Analyzing {len(chunks)} chunks for outlier content")

    vectors = []
    chunk_data = []

    for chunk in chunks:
        if chunk.get('vector'):
            vectors.append(chunk['vector'])
            chunk_data.append(chunk)

    if len(vectors) < 5:
        logging.warning(f"Not enough valid vectors for clustering: {len(vectors)}")
        return []

    X = np.array(vectors)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(X)

    outlier_indices = [i for i, label in enumerate(labels) if label == -1]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = len(outlier_indices)

    logging.debug(f"Found {n_clusters} clusters and {n_outliers} outliers")

    results = []
    for idx in outlier_indices:
        chunk = chunk_data[idx]

        distances = []
        for other_idx, other_vector in enumerate(vectors):
            if idx != other_idx:
                cos_sim = np.dot(X[idx], other_vector) / (np.linalg.norm(X[idx]) * np.linalg.norm(other_vector))
                distances.append(1 - cos_sim)

        avg_distance = np.mean(distances) if distances else 0

        results.append({
            "chunk_id": chunk["id"],
            "text": chunk["payload"].get("text", ""),
            "chunk_index": chunk["payload"].get("chunk_index", 0),
            "outlier_score": float(avg_distance),
            "cluster_label": -1
        })

    results.sort(key=lambda x: x["outlier_score"], reverse=True)

    return results
