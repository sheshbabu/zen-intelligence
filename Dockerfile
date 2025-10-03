FROM --platform=$BUILDPLATFORM python:3.12-slim-bookworm as builder

ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

FROM --platform=$TARGETPLATFORM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY commons ./commons
COPY features ./features
COPY main.py ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8001

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info", "--no-access-log"]
