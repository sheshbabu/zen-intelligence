init:
	uv sync

dev:
	uv run python main.py

# UI: http://localhost:6333/dashboard
qdrant:
	docker run -d --name zen-qdrant -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest

qdrant-down:
	docker stop zen-qdrant && docker rm zen-qdrant
