import logging
import numpy as np
import spacy
from typing import List
from unmarkd import unmark
from sklearn.metrics.pairwise import cosine_similarity
from commons.qdrant.qdrant_helper import embed_texts


CHUNK_BREAK_THRESHOLD = 0.5
MIN_SENTENCES_PER_CHUNK = 2
MAX_SENTENCES_PER_CHUNK = 10
MIN_CHUNK_LENGTH = 20

nlp = spacy.load("en_core_web_sm")


def chunk_note(content: str) -> List[str]:
    if not content.strip():
        return []

    content = unmark(content)
    content = content.strip()

    if not content:
        return []

    try:
        chunks = split_by_semantic_similarity(content)
        return chunks if chunks else []
    except Exception as e:
        logging.error(f"Chunking failed: {e}")
        return []


def split_by_semantic_similarity(content: str) -> List[str]:
    sentences = split_into_sentences(content)
    if len(sentences) <= 1:
        return [content]

    logging.debug(f"Starting semantic splitting with {len(sentences)} sentences")

    chunks = []
    current_chunk = []

    for i in range(len(sentences)):
        current_chunk.append(sentences[i])

        should_break_chunk = False

        if len(current_chunk) >= MAX_SENTENCES_PER_CHUNK:
            should_break_chunk = True

        if i < len(sentences) - 1:
            embeddings = embed_texts([sentences[i], sentences[i + 1]])
            similarity = cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[1]]))[0][0]
            if similarity < CHUNK_BREAK_THRESHOLD:
                should_break_chunk = True

        if i == len(sentences) - 1:
            should_break_chunk = True

        if not should_break_chunk:
            continue

        if len(current_chunk) >= MIN_SENTENCES_PER_CHUNK:
            chunk = ' '.join(current_chunk).strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
            current_chunk = []
            continue

        if len(current_chunk) < MIN_SENTENCES_PER_CHUNK and i == len(sentences) - 1:
            chunk = ' '.join(current_chunk).strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                if chunks:
                    chunks[-1] += ' ' + chunk
                else:
                    chunks.append(chunk)
            current_chunk = []

    if current_chunk:
        chunk = ' '.join(current_chunk).strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            if chunks and len(current_chunk) < MIN_SENTENCES_PER_CHUNK:
                chunks[-1] += ' ' + chunk
            else:
                chunks.append(chunk)

    logging.debug(f"Completed semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
    return chunks


def split_into_sentences(content: str) -> List[str]:
    doc = nlp(content)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences