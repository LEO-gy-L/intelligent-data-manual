from fastapi import FastAPI

app = FastAPI(title="Datasheet RAG v2")

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
