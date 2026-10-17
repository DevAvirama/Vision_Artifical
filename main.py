# Re-exporta la aplicación FastAPI desde visionArtificial.py
from visionArtificial import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
