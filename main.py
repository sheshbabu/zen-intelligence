import os
import uvicorn
from fastapi import FastAPI, HTTPException
from features.embedding.embedding_routes import router as embedding_router
from features.search.search_routes import router as search_router
from commons.qdrant.qdrant_client import health_check

app = FastAPI(title="Zen Intelligence", version="0.1.0")

app.include_router(embedding_router)
app.include_router(search_router)

@app.get("/health")
async def health():
    qdrant_healthy = health_check()
    if not qdrant_healthy:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Zen Intelligence API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")