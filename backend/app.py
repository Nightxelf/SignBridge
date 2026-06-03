from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes.predict import router as predict_router
from backend.utils.config import Settings

settings = Settings()

app = FastAPI(title="SignBridge API")
app.include_router(predict_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "SignBridge backend running", "host": settings.host, "port": settings.port}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
