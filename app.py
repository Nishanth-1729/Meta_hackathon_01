import os
import uvicorn
from crust_env.api import app

# Entry point for Hugging Face Spaces (FastAPI / Docker)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
