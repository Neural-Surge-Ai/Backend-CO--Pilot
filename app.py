from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import  chat_api


app = FastAPI(title="AI-Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Routers
#app.include_router(scrape_api.router)
app.include_router(chat_api.router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
