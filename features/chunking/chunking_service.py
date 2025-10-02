import re
import math
import logging
from typing import List
from commons.qdrant.qdrant_helper import embed_texts


CHUNK_BREAK_THRESHOLD = 0.5
MIN_SENTENCES_PER_CHUNK = 2
MAX_SENTENCES_PER_CHUNK = 10


markdown_url_regex = re.compile(r'!?\[([^\]]*)\]\([^)]+\)')
url_regex = re.compile(r'https?://[^\s]+')
code_fence_regex = re.compile(r'(?s)```[^`]*```')
inline_code_regex = re.compile(r'`[^`]+`')
sentence_regex = re.compile(r'[.?!]\s+')


def filter_urls(content: str) -> str:
    content = markdown_url_regex.sub(r'\1', content)
    content = url_regex.sub('', content)
    return content


def filter_code_blocks(content: str) -> str:
    content = code_fence_regex.sub('', content)
    content = inline_code_regex.sub('', content)
    return content


def split_into_sentences(content: str) -> List[str]:
    paragraphs = content.split('\n')
    all_sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = split_paragraph_into_sentences(paragraph)
        all_sentences.extend(sentences)

    return all_sentences


def split_paragraph_into_sentences(paragraph: str) -> List[str]:
    matches = list(sentence_regex.finditer(paragraph))

    if not matches:
        return [paragraph.strip()]

    sentences = []
    start = 0

    for match in matches:
        end = match.start() + 1
        sentence = paragraph[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()

    if start < len(paragraph):
        last_sentence = paragraph[start:].strip()
        if last_sentence:
            sentences.append(last_sentence)

    return sentences


def calc_cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def split_by_semantic_similarity(content: str) -> List[str]:
    sentences = split_into_sentences(content)
    if len(sentences) <= 1:
        return [content]

    logging.debug(f"Starting semantic splitting with {len(sentences)} sentences")

    embeddings = embed_texts(sentences)

    chunks = []
    current_chunk = []

    for i in range(len(sentences)):
        current_chunk.append(sentences[i])

        should_break_chunk = False

        if len(current_chunk) >= MAX_SENTENCES_PER_CHUNK:
            should_break_chunk = True

        if i < len(sentences) - 1:
            similarity = calc_cosine_similarity(embeddings[i], embeddings[i + 1])
            if similarity < CHUNK_BREAK_THRESHOLD:
                should_break_chunk = True

        if i == len(sentences) - 1:
            should_break_chunk = True

        if not should_break_chunk:
            continue

        if len(current_chunk) >= MIN_SENTENCES_PER_CHUNK:
            chunk = ' '.join(current_chunk).strip()
            chunks.append(chunk)
            current_chunk = []
            continue

        if len(current_chunk) < MIN_SENTENCES_PER_CHUNK and i == len(sentences) - 1:
            chunk = ' '.join(current_chunk).strip()
            if chunks:
                chunks[-1] += ' ' + chunk
            else:
                chunks.append(chunk)
            current_chunk = []

    if current_chunk:
        if chunks and len(current_chunk) < MIN_SENTENCES_PER_CHUNK:
            chunk = ' '.join(current_chunk).strip()
            chunks[-1] += ' ' + chunk
        else:
            chunk = ' '.join(current_chunk).strip()
            chunks.append(chunk)

    logging.debug(f"Completed semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
    return chunks


def chunk_note(content: str) -> List[str]:
    if not content.strip():
        return []

    content = filter_urls(content)
    content = filter_code_blocks(content)
    content = content.strip()

    if not content:
        return []

    try:
        chunks = split_by_semantic_similarity(content)
        return chunks if chunks else [content]
    except Exception as e:
        logging.error(f"Chunking failed: {e}")
        return [content]