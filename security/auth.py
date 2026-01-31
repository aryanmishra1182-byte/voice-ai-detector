import os
from fastapi import HTTPException

API_KEY = os.getenv("API_KEY")

def verify_api_key(api_key: str):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
