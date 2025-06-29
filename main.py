# main.py for demand_engine FastAPI application
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# Initialize the FastAPI application
app = FastAPI(
    title="Demand Engine Map Server",
    description="Serves HTML map files based on date and stockpoint.",
    version="1.0.0"
)

# Define the base directory for your map files.
# This path is absolute and points to where your HTML map files are stored.
MAPS_BASE_DIR = Path("/home/azureuser/BT/11_Demand_Engine/01_ALGO/Recommendation/routing/")

@app.get("/maps/{date}/{stockpoint_id}.html")
async def get_map_file(date: str, stockpoint_id: str):
    """
    Serves HTML map files based on the provided date and stockpoint ID.
    Example URL: /maps/2025-06-20/88999.html
    """
    # Construct the full path to the HTML file based on the URL parameters.
    # Example: /home/azureuser/BT/11_Demand_Engine/01_ALGO/Recommendation/routing/2025-06-20/88999.html
    file_path = MAPS_BASE_DIR / date / f"{stockpoint_id}.html"

    # Print the attempted file path to the console for debugging purposes.
    print(f"Attempting to serve file: {file_path}")

    # Check if the constructed file path exists and is a file.
    if not file_path.is_file():
        print(f"File not found: {file_path}") # Debugging print
        raise HTTPException(status_code=404, detail=f"Map file not found for date '{date}' and stockpoint '{stockpoint_id}'.")

    # Security check: Ensure the resolved file path is within the allowed base directory.
    # This prevents directory traversal attacks where users might try to access files
    # outside the intended `MAPS_BASE_DIR`.
    try:
        # Resolve the absolute path and check if it's a subpath of MAPS_BASE_DIR.
        file_path.resolve(strict=True).relative_to(MAPS_BASE_DIR.resolve(strict=True))
    except ValueError:
        print(f"Attempted path traversal detected for: {file_path}")
        raise HTTPException(status_code=400, detail="Invalid file path requested. Path traversal detected.")
    except FileNotFoundError:
        # This can happen if strict=True fails to resolve a non-existent file,
        # though it should ideally be caught by file_path.is_file() first.
        raise HTTPException(status_code=404, detail=f"Map file not found for date '{date}' and stockpoint '{stockpoint_id}'.")


    # Return the file as a FileResponse.
    # The media_type is crucial for browsers to correctly interpret the file as HTML.
    return FileResponse(file_path, media_type="text/html")

@app.get("/")
async def read_root():
    """
    Root endpoint for basic health check or information.
    """
    return {"message": "Demand Engine Map Server is running. Access maps via /maps/{date}/{stockpoint_id}.html\n maps/2025-06-23/1647033.html"}
