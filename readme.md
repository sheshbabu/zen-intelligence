# Zen Intelligence

Semantic intelligence service for Zen Notes

> **⚠️ Warning:** This repository is experimental and under active development. Expect breaking changes without notice.

### Setup

1. Install dependencies:
```bash
make init
```

2. Start Qdrant:
```bash
make qdrant
```

3. Run the service:
```bash
make dev
```

### Docker Compose

```yaml
services:
  zen:
    image: ghcr.io/sheshbabu/zen/zen:latest
    container_name: zen
    ports:
      - 8080:8080
    volumes:
      - ./data:/data
      - ./images:/images
    environment:
      - INTELLIGENCE_ENABLED=true
      - ZEN_INTELLIGENCE_URL=http://zen-intelligence:8001
    restart: 'unless-stopped'
    depends_on:
      - zen-intelligence

  zen-intelligence:
    image: ghcr.io/sheshbabu/zen-intelligence/zen-intelligence:latest
    container_name: zen-intelligence
    ports:
      - 8001:8001
    volumes:
      - ./images:/images
      - ./huggingface-cache:/root/.cache/huggingface
    environment:
      - QDRANT_URL=http://qdrant:6333
    restart: 'unless-stopped'
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - 6333:6333
    volumes:
      - qdrant-data:/qdrant/storage
    restart: 'unless-stopped'

volumes:
  qdrant-data:
```