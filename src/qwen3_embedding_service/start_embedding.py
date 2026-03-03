import uvicorn
from embedding_service import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)  # nosec B104
