import os
import logging
from typing import List
from fastembed import TextEmbedding, ImageEmbedding
from commons.qdrant.qdrant_client import create_collection_if_not_exists

NOTE_COLLECTION = "notes_v1"
IMAGE_COLLECTION = "images_v1"
TEXT_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
IMAGE_EMBED_MODEL = "Qdrant/clip-ViT-B-32-vision"
IMAGE_QUERY_MODEL = "Qdrant/clip-ViT-B-32-text"


create_collection_if_not_exists(NOTE_COLLECTION, 768)
create_collection_if_not_exists(IMAGE_COLLECTION, 512)


text_model = TextEmbedding(model_name=TEXT_EMBED_MODEL)
image_model = ImageEmbedding(model_name=IMAGE_EMBED_MODEL)
image_query_model = TextEmbedding(model_name=IMAGE_QUERY_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    embeddings = list(text_model.embed(texts))
    return [embedding.tolist() for embedding in embeddings]


def embed_text(text: str) -> List[float]:
    if not text.strip():
        raise ValueError("Text cannot be empty")

    return embed_texts([text])[0]


def embed_image(image_path: str) -> List[float]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    embeddings = list(image_model.embed([image_path]))
    return embeddings[0].tolist()


def embed_query_for_images(query: str) -> List[float]:
    embeddings = list(image_query_model.embed([query]))
    return embeddings[0].tolist()