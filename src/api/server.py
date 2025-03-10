import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    reload = os.getenv("ENV", "production").lower() == "development"
    port = int(os.getenv("PORT", 8000))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        workers=4 if not reload else 1,
        # workers=1,
        reload=reload,
        reload_dirs=[project_root + "/api/src"],
    )