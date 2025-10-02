import os
import logging
from typing import List, Dict, Any
from qdrant_client.qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue)


url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=url, timeout=0.5)


def create_collection_if_not_exists(collection_name: str, vector_size: int) -> None:
    collections = client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if collection_name not in existing_names:
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
        logging.info(f"Created collection: {collection_name}")


def upsert_points(collection_name: str, points: List[Dict[str, Any]]) -> None:
    point_structs = [
        PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
        for point in points
    ]
    client.upsert(collection_name=collection_name, points=point_structs)
    logging.debug(f"Upserted {len(points)} points to {collection_name}")


def search_similar(collection_name: str, query_vector: List[float], limit: int = 20, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    results = client.search(collection_name=collection_name, query_vector=query_vector, limit=limit, score_threshold=score_threshold)
    return [
        {
            "id": str(result.id),
            "score": float(result.score),
            "payload": result.payload or {},
        }
        for result in results
    ]


def delete_points_by_filter(collection_name: str, filter_conditions: Dict[str, Any]) -> None:
    points_to_delete = scroll_points(collection_name, filter_conditions, limit=1000)

    if not points_to_delete:
        return

    point_ids = [point["id"] for point in points_to_delete]
    client.delete(collection_name=collection_name, points_selector=point_ids)
    logging.debug(f"Deleted {len(point_ids)} points from {collection_name}")


def scroll_points(collection_name: str, filter_conditions: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
    search_filter = Filter(must=[
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in filter_conditions.items()
    ])

    results, _ = client.scroll(collection_name=collection_name, scroll_filter=search_filter, limit=limit, with_payload=True)

    return [
        {"id": str(result.id), "payload": result.payload or {}}
        for result in results
    ]


def health_check() -> bool:
    try:
        client.get_collections()
        return True
    except Exception as e:
        logging.error(f"qdrant health check failed: {e}")
        return False