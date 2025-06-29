# # main.py
# from fastapi import FastAPI, HTTPException
# from starlette.responses import FileResponse
# from pathlib import Path
# import os

# app = FastAPI()

# # Define the absolute base directory for your map files
# # This ensures the path is always correct regardless of where uvicorn is launched from
# MAPS_BASE_DIR = Path("/home/azureuser/BT/11_Demand_Engine/01_ALGO/Recommendation/routing")

# @app.on_event("startup")
# async def startup_event():
#     # Ensure the maps base directory exists at startup
#     if not MAPS_BASE_DIR.is_dir():
#         print(f"Warning: Map base directory not found: {MAPS_BASE_DIR}. Please ensure this directory exists.")
#         # Optionally, you could raise an error or create it, depending on your needs
#         # raise RuntimeError(f"Map base directory not found: {MAPS_BASE_DIR}")

# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the FastAPI Map Server!"}

# @app.get("/maps/{date_dir}/{stockpoint_id}.html")
# async def serve_map_html(date_dir: str, stockpoint_id: str):
#     """
#     Serves HTML map files based on date directory and stockpoint ID.
#     Example path: /maps/2025-06-20/88999.html
#     """
#     # Basic validation for date_dir to prevent directory traversal
#     # Ensure no '..' or '/' or '\' in the date_dir segment
#     if ".." in date_dir or "/" in date_dir or "\\" in date_dir:
#         raise HTTPException(status_code=400, detail="Invalid date directory format.")

#     # Construct the full path to the HTML file
#     requested_file_path = MAPS_BASE_DIR / date_dir / f"{stockpoint_id}.html"

#     # Log the path for debugging
#     print(f"Attempting to serve: {requested_file_path}")

#     # Security check: Ensure the resolved path is within the intended base directory
#     # Use .resolve() on both paths to handle symlinks and canonical paths
#     try:
#         resolved_requested_file_path = requested_file_path.resolve()
#         resolved_maps_base_dir = MAPS_BASE_DIR.resolve()

#         print(f"Resolved Requested Path: {resolved_requested_file_path}")
#         print(f"Resolved Base Path: {resolved_maps_base_dir}")

#         # --- MODIFICATION START ---
#         # Replace .is_relative_to() for compatibility with Python < 3.9
#         # Check if the resolved requested path starts with the resolved base path
#         if not str(resolved_requested_file_path).startswith(str(resolved_maps_base_dir)):
#             print(f"Security Warning: Attempted access outside base directory.")
#             raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory.")
#         # --- MODIFICATION END ---

#     except FileNotFoundError:
#         # If .resolve() fails because the requested_file_path or its parent directories don't exist,
#         # it will raise FileNotFoundError. This is not a security issue, but a file not found.
#         # We will let the .is_file() check handle the 404.
#         print(f"Path component not found during resolve: {requested_file_path}")
#         pass # Continue to is_file() check
#     except Exception as e: # Catch a broader exception for unexpected issues during path resolution
#         print(f"Error during path resolution/relative check: {e}")
#         # Provide more detailed info for debugging
#         try:
#             resolved_req_path_info = requested_file_path.resolve() if requested_file_path.exists() else 'Does not exist or cannot resolve'
#         except Exception:
#             resolved_req_path_info = 'Error resolving requested path'
#         try:
#             resolved_base_path_info = MAPS_BASE_DIR.resolve() if MAPS_BASE_DIR.exists() else 'Does not exist or cannot resolve'
#         except Exception:
#             resolved_base_path_info = 'Error resolving base path'

#         print(f"Resolved Requested Path (at error): {resolved_req_path_info}")
#         print(f"Resolved Base Path (at error): {resolved_base_path_info}")
#         raise HTTPException(status_code=500, detail="Internal server error: Path validation failed unexpectedly.")


#     if not requested_file_path.is_file():
#         print(f"File not found: {requested_file_path}")
#         raise HTTPException(status_code=404, detail="Map not found.")

#     return FileResponse(requested_file_path, media_type="text/html")

# # This FastAPI application includes:
# # A root endpoint (/) for a welcome message.
# # A dynamic endpoint /maps/{date_dir}/{stockpoint_id}.html that:
# # Takes date_dir (e.g., 2025-06-20) and stockpoint_id (e.g., 88999) as path parameters.
# # Constructs the full file path within your 01_ALGO/Recommendation/routing directory.

# # Performs security checks to prevent directory traversal.

# # Returns the HTML file using FileResponse if found, otherwise returns a 404 error.



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# main.py
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pathlib import Path
import os

app = FastAPI()

# Define the absolute base directory for your map files
# This ensures the path is always correct regardless of where uvicorn is launched from
MAPS_BASE_DIR = Path("/home/azureuser/BT/11_Demand_Engine/01_ALGO/Recommendation/routing")

@app.on_event("startup")
async def startup_event():
    # Ensure the maps base directory exists at startup
    if not MAPS_BASE_DIR.is_dir():
        print(f"Warning: Map base directory not found: {MAPS_BASE_DIR}. Please ensure this directory exists.")
        # Optionally, you could raise an error or create it, depending on your needs
        # raise RuntimeError(f"Map base directory not found: {MAPS_BASE_DIR}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI Map Server!"}

@app.get("/maps/{date_dir}/{stockpoint_id}.html")
async def serve_map_html(date_dir: str, stockpoint_id: str):
    """
    Serves HTML map files based on date directory and stockpoint ID.
    Example path: /maps/2025-06-20/88999.html
    """
    # Basic validation for date_dir to prevent directory traversal
    # Ensure no '..' or '/' or '\' in the date_dir segment
    if ".." in date_dir or "/" in date_dir or "\\" in date_dir:
        raise HTTPException(status_code=400, detail="Invalid date directory format.")

    # Construct the full path to the HTML file
    requested_file_path = MAPS_BASE_DIR / date_dir / f"{stockpoint_id}.html"

    # Log the path for debugging
    print(f"Attempting to serve: {requested_file_path}")

    # Security check: Ensure the resolved path is within the intended base directory
    # Use .resolve() on both paths to handle symlinks and canonical paths
    try:
        resolved_requested_file_path = requested_file_path.resolve()
        resolved_maps_base_dir = MAPS_BASE_DIR.resolve()

        print(f"Resolved Requested Path: {resolved_requested_file_path}")
        print(f"Resolved Base Path: {resolved_maps_base_dir}")

        # Replace .is_relative_to() for compatibility with Python < 3.9
        # Check if the resolved requested path starts with the resolved base path
        if not str(resolved_requested_file_path).startswith(str(resolved_maps_base_dir)):
            print(f"Security Warning: Attempted access outside base directory.")
            raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directory.")

    except FileNotFoundError:
        # If .resolve() fails because the requested_file_path or its parent directories don't exist,
        # it will raise FileNotFoundError. This is not a security issue, but a file not found.
        # We will let the .is_file() check handle the 404.
        print(f"Path component not found during resolve: {requested_file_path}")
        pass # Continue to is_file() check
    except Exception as e: # Catch a broader exception for unexpected issues during path resolution
        print(f"Error during path resolution/relative check: {e}")
        # Provide more detailed info for debugging
        try:
            resolved_req_path_info = requested_file_path.resolve() if requested_file_path.exists() else 'Does not exist or cannot resolve'
        except Exception:
            resolved_req_path_info = 'Error resolving requested path'
        try:
            resolved_base_path_info = MAPS_BASE_DIR.resolve() if MAPS_BASE_DIR.exists() else 'Does not exist or cannot resolve'
        except Exception:
            resolved_base_path_info = 'Error resolving base path'

        print(f"Resolved Requested Path (at error): {resolved_req_path_info}")
        print(f"Resolved Base Path (at error): {resolved_base_path_info}")
        raise HTTPException(status_code=500, detail="Internal server error: Path validation failed unexpectedly.")


    if not requested_file_path.is_file():
        print(f"File not found: {requested_file_path}")
        raise HTTPException(status_code=404, detail="Map not found.")

    return FileResponse(requested_file_path, media_type="text/html")
