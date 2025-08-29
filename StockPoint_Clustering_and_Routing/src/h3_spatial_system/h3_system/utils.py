import h3
from shapely.geometry import Polygon
import logging

# Optional: configure logging if not already done
logging.basicConfig(level=logging.WARNING)

def h3_to_shapely_polygon(h3_index):
    """
    Converts an H3 index to a Shapely Polygon object.

    Args:
        h3_index (str): The H3 cell index.

    Returns:
        shapely.geometry.Polygon or None: The corresponding Shapely polygon,
        or None if an error occurs.
    """
    try:
        # 1. Get the boundary as a list of (lat, lng) tuples
        h3_boundary = h3.cell_to_boundary(h3_index)

        # 2. Swap coordinates to (lng, lat) for Shapely
        shapely_coords = [(lng, lat) for lat, lng in h3_boundary]

        # 3. Create and return the Shapely Polygon
        return Polygon(shapely_coords)

    except Exception as e:
        logging.warning(f"Failed to convert H3 index '{h3_index}' to Shapely Polygon: {e}")
        return None



from multiprocessing import Pool, cpu_count
import time
from typing import List

def convert_h3_cells_to_polygons_in_parallel(h3_cells: List[str]) -> List[Polygon]:
    """
    Converts a list of H3 cell indices to Shapely Polygons using multiprocessing.

    Args:
        h3_cells (List[str]): A list of H3 cell indices.

    Returns:
        List[Polygon]: A list of Shapely Polygon objects.
    """
    # --- Parallel (Accelerated) Method ---
    num_processes = cpu_count() - 1  # Determine the number of CPU cores to use.
    print(f"\nUsing {num_processes} CPU cores for parallel processing...")

    start_time = time.time()
    with Pool(num_processes) as pool:
        # Pool.map() applies the function to all items in parallel
        h3_cells_polygons_parallel = pool.map(h3_to_shapely_polygon, h3_cells)
    parallel_time = time.time() - start_time

    print(f"Parallel conversion of {len(h3_cells):,} cells took: {parallel_time:.2f} seconds")

    return h3_cells_polygons_parallel   



# ---------------------------------------------
# ---------------------------------------------
# ---------------------------------------------
from shapely.geometry import Polygon
import h3
from typing import List, Dict, Any, Tuple, Iterator, Optional
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp

def h3_to_objects(h3_index: str) -> Dict[str, Any]:
    """
    Transforms an H3 cell index into a structured dictionary containing geometric data.

    This function is optimized for performance by minimizing calls to the H3 library
    and processing coordinates in a single loop. It's ideal for bulk operations
    where a comprehensive set of geometric properties (boundary, centroid, coordinates,
    and a Shapely polygon) is required for a given H3 index.

    Args:
        h3_index (str): A string representing a valid H3 cell index.

    Returns:
        dict: A dictionary containing the following keys:
            - 'h3_index' (str): The original H3 index.
            - 'centroid' (tuple | None): A tuple of (latitude, longitude) for the cell's
              center, or None if an error occurs.
            - 'boundary' (list | None): A list of (latitude, longitude) tuples
              defining the cell's boundary, or None.
            - 'latlng_coords' (list | None): A list of lists, where each inner list
              is [latitude, longitude], representing the cell's vertices, or None.
            - 'polygon' (Polygon | None): A Shapely Polygon [longitude, latitude] object representing
              the cell's shape, or None.
            - 'error' (str, optional): A descriptive error message if an exception
              is caught, otherwise not present.
    """
    try:
        # Get boundary once - this is the most expensive operation
        boundary = h3.cell_to_boundary(h3_index)
        
        # Process coordinates in single loop to minimize iterations
        latlng_coords = []
        shapely_coords = []
        
        for lat, lng in boundary:
            latlng_coords.append([lat, lng])
            shapely_coords.append((lng, lat))  # Shapely uses (lng, lat)
        
        return {
            "h3_index": h3_index,
            "centroid": h3.cell_to_latlng(h3_index),
            "boundary": boundary,
            "latlng_coords": latlng_coords,
            "polygon": Polygon(shapely_coords)
        }
    except Exception as e:
        return {
            "h3_index": h3_index,
            "centroid": None,
            "boundary": None, 
            "latlng_coords": None,
            "polygon": None,
            "error": str(e)
        }


def is_jupyter_environment():
    """Check if running in Jupyter notebook/lab."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def h3_to_objects_parallel_safe(h3_indices: List[str], n_workers: Optional[int] = None, 
                               use_threads: Optional[bool] = None) -> Dict[str, Dict[str, Any]]:
    """
    Jupyter-safe parallel processing of H3 indices.
    Automatically chooses between ProcessPoolExecutor and ThreadPoolExecutor.
    
    Args:
        h3_indices: List of H3 cell indices to process
        n_workers: Number of workers to use. Auto-detected if None.
        use_threads: Force thread/process usage. Auto-detected if None.
        
    Returns:
        Dict with H3 indices as keys and their data as values
    """
    if not h3_indices:
        return {}
    
    # Auto-detect best approach
    if use_threads is None:
        use_threads = is_jupyter_environment()
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(h3_indices), 8)  # Cap at 8 to avoid overhead
    
    # Choose executor based on environment
    if use_threads:
        # print(f"Using ThreadPoolExecutor with {n_workers} threads (Jupyter-safe)")
        executor_class = ThreadPoolExecutor
    else:
        print(f"Using ProcessPoolExecutor with {n_workers} processes")
        executor_class = ProcessPoolExecutor
    
    results = {}
    
    try:
        with executor_class(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_h3 = {executor.submit(h3_to_objects, h3): h3 for h3 in h3_indices}
            
            # Collect results as they complete
            for future in as_completed(future_to_h3):
                result = future.result()
                results[result["h3_index"]] = result
                
        return results
        
    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to sequential")
        return h3_to_objects_sequential(h3_indices)


def h3_to_objects_sequential(h3_indices: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Sequential processing - always works, good for small datasets or debugging.
    """
    return {h3_index: h3_to_objects(h3_index) for h3_index in h3_indices}


def h3_to_objects_generator(h3_indices: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Memory-optimized generator - processes one at a time.
    """
    for h3_index in h3_indices:
        result = h3_to_objects(h3_index)
        yield h3_index, result


def h3_to_objects_parallel_generator(h3_indices: List[str], n_workers: Optional[int] = None,
                                   batch_size: int = 100) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Parallel generator that processes in batches and yields results.
    More memory-efficient for very large datasets.
    
    Args:
        h3_indices: List of H3 indices
        n_workers: Number of workers
        batch_size: Process this many at a time
    """
    # Process in batches to control memory usage
    for i in range(0, len(h3_indices), batch_size):
        batch = h3_indices[i:i + batch_size]
        batch_results = h3_to_objects_parallel_safe(batch, n_workers)
        
        for h3_index, result in batch_results.items():
            yield h3_index, result


def process_h3_indices(h3_indices: List[str], 
                      method: str = "auto",
                      n_workers: Optional[int] = None,
                      **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Smart processing with automatic method selection.
    
    Args:
        h3_indices: List of H3 indices to process
        method: "auto", "sequential", "parallel", "generator", "parallel_generator"
        n_workers: Number of workers for parallel methods
        **kwargs: Additional arguments
        
    Returns:
        Dict with H3 indices as keys and their data as values
    """
    num_indices = len(h3_indices)
    
    if method == "auto":
        if num_indices < 50:
            method = "sequential"
        elif num_indices < 5000:
            method = "parallel"
        else:
            method = "parallel_generator"
    
    print(f"Processing {num_indices} H3 indices using {method} method")
    
    if method == "sequential":
        return h3_to_objects_sequential(h3_indices)
    elif method == "parallel":
        return h3_to_objects_parallel_safe(h3_indices, n_workers)
    elif method == "generator":
        return {h3_index: result for h3_index, result in h3_to_objects_generator(h3_indices)}
    elif method == "parallel_generator":
        return {h3_index: result for h3_index, result in h3_to_objects_parallel_generator(h3_indices, n_workers, **kwargs)}
    else:
        raise ValueError(f"Unknown method: {method}")


# Jupyter-specific helper
def process_h3_in_jupyter(h3_indices: List[str], show_progress: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Jupyter-optimized processing with optional progress bar.
    """
    if show_progress:
        try:
            from tqdm.auto import tqdm
            
            def h3_with_progress(h3_list):
                for h3_index in tqdm(h3_list, desc="Processing H3 cells"):
                    yield h3_index, h3_to_objects(h3_index)
            
            return {h3: result for h3, result in h3_with_progress(h3_indices)}
            
        except ImportError:
            print("Install tqdm for progress bar: pip install tqdm")
    
    # Fallback to regular processing
    return process_h3_indices(h3_indices, method="auto")


if __name__ == "__main__":
    # Test with sample data
    sample_h3_indices = [
        "8928308280fffff",
        "8928308280bffff", 
        "89283082807ffff",
        "89283082803ffff"
    ]
    
    print("=== H3 Processing Tests ===")
    
    # Test 1: Sequential (always safe)
    print("\n1. Sequential Processing:")
    results = h3_to_objects_sequential(sample_h3_indices)
    for h3_index in list(results.keys())[:2]:  # Show first 2
        centroid = results[h3_index].get('centroid', 'Error')
        print(f"   {h3_index} -> {centroid}")
    
    # Test 2: Parallel (Jupyter-safe)
    print("\n2. Parallel Processing (Jupyter-safe):")
    results = h3_to_objects_parallel_safe(sample_h3_indices)
    for h3_index in list(results.keys())[:2]:  # Show first 2
        centroid = results[h3_index].get('centroid', 'Error')
        print(f"   {h3_index} -> {centroid}")
    
    # Test 3: Auto-selection
    print("\n3. Auto-selection:")
    results = process_h3_indices(sample_h3_indices)
    print(f"   Processed {len(results)} H3 cells successfully")
    
    # Test 4: Dictionary access
    print("\n4. Dictionary Access:")
    sample_h3 = sample_h3_indices[0]
    if sample_h3 in results and 'centroid' in results[sample_h3]:
        lat, lng = results[sample_h3]['centroid']
        print(f"   H3 {sample_h3}:")
        print(f"     Centroid: ({lat:.6f}, {lng:.6f})")
        print(f"     Polygon area: {results[sample_h3]['polygon'].area:.10f}")
    
    print("\n=== Usage Recommendations ===")
    print("For Jupyter notebooks:")
    print("  results = process_h3_indices(h3_list)  # Auto-detects best method")
    print("  results = process_h3_in_jupyter(h3_list, show_progress=True)  # With progress bar")
    print("\nFor scripts:")
    print("  results = h3_to_objects_parallel_safe(h3_list, use_threads=False)  # Use processes")

import json
import os
from pathlib import Path
from typing import Dict, Any

def save_to_json_efficient(h3_index: str, result: Dict[str, Any], file_path: str):
    """
    Efficient JSON saver that doesn't rewrite the entire file each time.
    Uses JSONL (JSON Lines) format for streaming writes.
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Shapely polygon to coordinates for JSON serialization
    if 'polygon' in result and result['polygon'] and 'error' not in result:
        result_copy = result.copy()
        result_copy['polygon_coords'] = list(result['polygon'].exterior.coords)
        del result_copy['polygon']  # Remove non-serializable Shapely object
        result_to_save = result_copy
    else:
        result_to_save = {k: v for k, v in result.items() if k != 'polygon'}
    
    # Append one JSON object per line (JSONL format)
    with open(file_path, 'a') as f:
        json.dump(result_to_save, f)
        f.write('\n')


def save_to_json_batch(h3_data: Dict[str, Dict[str, Any]], file_path: str):
    """
    Save all H3 data at once to a single JSON file.
    More efficient for final output.
    """
    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert all Shapely polygons to coordinates
    serializable_data = {}
    for h3_index, result in h3_data.items():
        if 'polygon' in result and result['polygon'] and 'error' not in result:
            result_copy = result.copy()
            result_copy['polygon_coords'] = list(result['polygon'].exterior.coords)
            del result_copy['polygon']
            serializable_data[h3_index] = result_copy
        else:
            serializable_data[h3_index] = {k: v for k, v in result.items() if k != 'polygon'}
    
    # Save as single JSON file
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load JSONL file back into dictionary.
    """
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    data[result['h3_index']] = result
    return data


# Alternative: Collect in batches then save
class H3JSONBatchSaver:
    """
    Collects H3 results and saves in batches to avoid memory issues
    while still being more efficient than one-by-one JSON saves.
    """
    
    def __init__(self, file_path: str, batch_size: int = 1000):
        self.file_path = file_path
        self.batch_size = batch_size
        self.batch_data = {}
        self.total_saved = 0
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Clear the file at start
        open(file_path, 'w').close()
    
    def add_result(self, h3_index: str, result: Dict[str, Any]):
        """Add a result to the current batch."""
        # Convert polygon for JSON serialization
        if 'polygon' in result and result['polygon'] and 'error' not in result:
            result_copy = result.copy()
            result_copy['polygon_coords'] = list(result['polygon'].exterior.coords)
            del result_copy['polygon']
            self.batch_data[h3_index] = result_copy
        else:
            self.batch_data[h3_index] = {k: v for k, v in result.items() if k != 'polygon'}
        
        # Save batch if it's full
        if len(self.batch_data) >= self.batch_size:
            self._save_batch()
    
    def _save_batch(self):
        """Save current batch to JSONL file."""
        if not self.batch_data:
            return
            
        with open(self.file_path, 'a') as f:
            for h3_index, result in self.batch_data.items():
                json.dump(result, f)
                f.write('\n')
        
        self.total_saved += len(self.batch_data)
        print(f"Saved batch of {len(self.batch_data)} items (total: {self.total_saved})")
        self.batch_data.clear()
    
    def finalize(self):
        """Save any remaining data in the batch."""
        if self.batch_data:
            self._save_batch()
        print(f"Finished! Total items saved: {self.total_saved}")


# Usage examples:
if __name__ == "__main__":
    # Example usage patterns
    
    print("=== H3 JSON Saving Examples ===")
    
    # Example 1: Stream to JSONL (most memory efficient)
    print("\n1. Streaming to JSONL:")
    """
    for h3_index, result in h3_to_objects_parallel_generator(h3_list):
        save_to_json_efficient(h3_index, result, "./output/processed/h3_data.jsonl")
    """
    
    # Example 2: Batch saving (balanced approach)
    print("2. Batch saving:")
    """
    saver = H3JSONBatchSaver("./output/processed/h3_data_batched.jsonl", batch_size=500)
    
    for h3_index, result in h3_to_objects_parallel_generator(h3_list):
        saver.add_result(h3_index, result)
    
    saver.finalize()  # Don't forget this!
    """
    
    # Example 3: All at once (if you have enough memory)
    print("3. All at once:")
    """
    # Process all data first
    h3_data = process_h3_indices(h3_list)
    
    # Save as single JSON file
    save_to_json_batch(h3_data, "./output/processed/h3_data.json")
    """
    
    print("\n=== Recommendations ===")
    print("• Small datasets (<1000): Use save_to_json_batch()")
    print("• Medium datasets (1000-10000): Use H3JSONBatchSaver")
    print("• Large datasets (>10000): Use save_to_json_efficient() with JSONL")
    print("• Load JSONL back with: data = load_jsonl('file.jsonl')")


# ----------------------------
# ----------------------------
# Save data utils
# ----------------------------
# ----------------------------
import sqlite3
import pickle
import zlib
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator
from shapely.geometry import Polygon
import h3

class H3SQLiteManager:
    """
    Complete SQLite-based storage solution for H3 data.
    Provides efficient storage, retrieval, and querying capabilities.
    """
    
    def __init__(self, db_path: str, batch_size: int = 1000):
        """
        Initialize H3 SQLite manager.
        
        Args:
            db_path: Path to SQLite database file
            batch_size: Number of records to batch before committing
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.batch_count = 0
        self.total_saved = 0
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._create_indexes()
    
    def _create_tables(self):
        """Create the H3 data table with optimized schema."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS h3_cells (
                h3_index TEXT PRIMARY KEY,
                resolution INTEGER,
                centroid_lat REAL,
                centroid_lng REAL,
                polygon_coords BLOB,
                boundary_data BLOB,
                latlng_coords BLOB,
                polygon_area REAL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def _create_indexes(self):
        """Create indexes for faster querying."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_resolution ON h3_cells(resolution)",
            "CREATE INDEX IF NOT EXISTS idx_centroid_lat ON h3_cells(centroid_lat)",
            "CREATE INDEX IF NOT EXISTS idx_centroid_lng ON h3_cells(centroid_lng)",
            "CREATE INDEX IF NOT EXISTS idx_area ON h3_cells(polygon_area)",
            "CREATE INDEX IF NOT EXISTS idx_error ON h3_cells(error)"
        ]
        
        for index_sql in indexes:
            self.cursor.execute(index_sql)
        self.conn.commit()
    
    def add_result(self, h3_index: str, result: Dict[str, Any]) -> None:
        """
        Add a single H3 result to the database.
        
        Args:
            h3_index: H3 cell index
            result: Result dictionary from h3_to_objects()
        """
        try:
            resolution = h3.h3_get_resolution(h3_index)
        except:
            resolution = None
        
        if 'error' in result:
            # Handle error cases
            self.cursor.execute('''
                INSERT OR REPLACE INTO h3_cells 
                (h3_index, resolution, error)
                VALUES (?, ?, ?)
            ''', (h3_index, resolution, result['error']))
        else:
            # Compress and store successful results
            polygon_coords = zlib.compress(
                pickle.dumps(list(result['polygon'].exterior.coords), protocol=pickle.HIGHEST_PROTOCOL)
            )
            boundary_data = zlib.compress(
                pickle.dumps(result['boundary'], protocol=pickle.HIGHEST_PROTOCOL)
            )
            latlng_coords = zlib.compress(
                pickle.dumps(result['latlng_coords'], protocol=pickle.HIGHEST_PROTOCOL)
            )
            
            lat, lng = result['centroid']
            area = result['polygon'].area
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO h3_cells 
                (h3_index, resolution, centroid_lat, centroid_lng, 
                 polygon_coords, boundary_data, latlng_coords, polygon_area)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (h3_index, resolution, lat, lng, polygon_coords, boundary_data, latlng_coords, area))
        
        self.batch_count += 1
        self.total_saved += 1
        
        # Commit in batches for better performance
        if self.batch_count >= self.batch_size:
            self.conn.commit()
            # print(f"Saved batch (total: {self.total_saved} H3 cells)")
            self.batch_count = 0
    
    def add_batch(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Add multiple H3 results efficiently.
        
        Args:
            results: Dictionary of {h3_index: result_dict}
        """
        data_to_insert = []
        
        for h3_index, result in results.items():
            try:
                resolution = h3.h3_get_resolution(h3_index)
            except:
                resolution = None
            
            if 'error' in result:
                data_to_insert.append((h3_index, resolution, None, None, None, None, None, None, result['error']))
            else:
                # Compress data
                polygon_coords = zlib.compress(
                    pickle.dumps(list(result['polygon'].exterior.coords), protocol=pickle.HIGHEST_PROTOCOL)
                )
                boundary_data = zlib.compress(
                    pickle.dumps(result['boundary'], protocol=pickle.HIGHEST_PROTOCOL)
                )
                latlng_coords = zlib.compress(
                    pickle.dumps(result['latlng_coords'], protocol=pickle.HIGHEST_PROTOCOL)
                )
                
                lat, lng = result['centroid']
                area = result['polygon'].area
                
                data_to_insert.append((
                    h3_index, resolution, lat, lng, 
                    polygon_coords, boundary_data, latlng_coords, area, None
                ))
        
        self.cursor.executemany('''
            INSERT OR REPLACE INTO h3_cells 
            (h3_index, resolution, centroid_lat, centroid_lng, 
             polygon_coords, boundary_data, latlng_coords, polygon_area, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        self.total_saved += len(data_to_insert)
        self.conn.commit()
        print(f"Saved batch of {len(data_to_insert)} items (total: {self.total_saved})")
    
    def get_result(self, h3_index: str) -> Optional[Dict[str, Any]]:
        """
        Get a single H3 result from the database.
        
        Args:
            h3_index: H3 cell index to retrieve
            
        Returns:
            Dictionary with H3 data or None if not found
        """
        self.cursor.execute('SELECT * FROM h3_cells WHERE h3_index = ?', (h3_index,))
        row = self.cursor.fetchone()
        
        if not row:
            return None
        
        (h3_idx, resolution, lat, lng, poly_coords, boundary, latlng, area, error, created_at) = row
        
        if error:
            return {
                "h3_index": h3_idx,
                "resolution": resolution,
                "error": error,
                "created_at": created_at
            }
        
        # Decompress data
        polygon_coords = pickle.loads(zlib.decompress(poly_coords))
        boundary_data = pickle.loads(zlib.decompress(boundary))
        latlng_data = pickle.loads(zlib.decompress(latlng))
        
        return {
            "h3_index": h3_idx,
            "resolution": resolution,
            "centroid": (lat, lng),
            "polygon": Polygon(polygon_coords),
            "boundary": boundary_data,
            "latlng_coords": latlng_data,
            "polygon_area": area,
            "created_at": created_at
        }
    
    def get_multiple(self, h3_indices: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple H3 results efficiently.
        
        Args:
            h3_indices: List of H3 cell indices
            
        Returns:
            Dictionary of {h3_index: result_dict}
        """
        placeholders = ','.join('?' * len(h3_indices))
        query = f'SELECT * FROM h3_cells WHERE h3_index IN ({placeholders})'
        
        self.cursor.execute(query, h3_indices)
        rows = self.cursor.fetchall()
        
        results = {}
        for row in rows:
            (h3_idx, resolution, lat, lng, poly_coords, boundary, latlng, area, error, created_at) = row
            
            if error:
                results[h3_idx] = {
                    "h3_index": h3_idx,
                    "resolution": resolution,
                    "error": error,
                    "created_at": created_at
                }
            else:
                # Decompress data
                polygon_coords = pickle.loads(zlib.decompress(poly_coords))
                boundary_data = pickle.loads(zlib.decompress(boundary))
                latlng_data = pickle.loads(zlib.decompress(latlng))
                
                results[h3_idx] = {
                    "h3_index": h3_idx,
                    "resolution": resolution,
                    "centroid": (lat, lng),
                    "polygon": Polygon(polygon_coords),
                    "boundary": boundary_data,
                    "latlng_coords": latlng_data,
                    "polygon_area": area,
                    "created_at": created_at
                }
        
        return results
    
    def query_by_resolution(self, resolution: int) -> List[str]:
        """Get all H3 indices for a specific resolution."""
        self.cursor.execute('SELECT h3_index FROM h3_cells WHERE resolution = ?', (resolution,))
        return [row[0] for row in self.cursor.fetchall()]
    
    def query_by_area_range(self, min_area: float, max_area: float) -> List[str]:
        """Get H3 indices within an area range."""
        self.cursor.execute('''
            SELECT h3_index FROM h3_cells 
            WHERE polygon_area BETWEEN ? AND ? AND error IS NULL
        ''', (min_area, max_area))
        return [row[0] for row in self.cursor.fetchall()]
    
    def query_by_bbox(self, min_lat: float, max_lat: float, min_lng: float, max_lng: float) -> List[str]:
        """Get H3 indices within a bounding box."""
        self.cursor.execute('''
            SELECT h3_index FROM h3_cells 
            WHERE centroid_lat BETWEEN ? AND ? 
            AND centroid_lng BETWEEN ? AND ?
            AND error IS NULL
        ''', (min_lat, max_lat, min_lng, max_lng))
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        # Total records
        self.cursor.execute('SELECT COUNT(*) FROM h3_cells')
        stats['total_records'] = self.cursor.fetchone()[0]
        
        # Records by resolution
        self.cursor.execute('''
            SELECT resolution, COUNT(*) 
            FROM h3_cells 
            WHERE error IS NULL 
            GROUP BY resolution
        ''')
        stats['by_resolution'] = dict(self.cursor.fetchall())
        
        # Error count
        self.cursor.execute('SELECT COUNT(*) FROM h3_cells WHERE error IS NOT NULL')
        stats['error_count'] = self.cursor.fetchone()[0]
        
        # Area statistics
        self.cursor.execute('''
            SELECT MIN(polygon_area), MAX(polygon_area), AVG(polygon_area)
            FROM h3_cells WHERE error IS NULL
        ''')
        min_area, max_area, avg_area = self.cursor.fetchone()
        stats['area_stats'] = {
            'min': min_area,
            'max': max_area,
            'average': avg_area
        }
        
        # File size
        stats['file_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        
        return stats
    
    def export_to_dict(self, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Export all data back to dictionary format.
        
        Args:
            limit: Maximum number of records to export
            
        Returns:
            Dictionary of {h3_index: result_dict}
        """
        if limit:
            self.cursor.execute('SELECT * FROM h3_cells LIMIT ?', (limit,))
        else:
            self.cursor.execute('SELECT * FROM h3_cells')
        
        results = {}
        for row in self.cursor.fetchall():
            (h3_idx, resolution, lat, lng, poly_coords, boundary, latlng, area, error, created_at) = row
            
            if error:
                results[h3_idx] = {
                    "h3_index": h3_idx,
                    "resolution": resolution,
                    "error": error
                }
            else:
                # Decompress data
                polygon_coords = pickle.loads(zlib.decompress(poly_coords))
                boundary_data = pickle.loads(zlib.decompress(boundary))
                latlng_data = pickle.loads(zlib.decompress(latlng))
                
                results[h3_idx] = {
                    "h3_index": h3_idx,
                    "resolution": resolution,
                    "centroid": (lat, lng),
                    "polygon": Polygon(polygon_coords),
                    "boundary": boundary_data,
                    "latlng_coords": latlng_data,
                    "polygon_area": area
                }
        
        return results
    
    def finalize(self):
        """Commit any remaining transactions."""
        if self.batch_count > 0:
            self.conn.commit()
            print(f"Final commit: {self.total_saved} total H3 cells saved")
    
    def close(self):
        """Close database connection."""
        self.finalize()
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Integration with your existing H3 processing functions
def save_h3_data_to_sqlite(h3_indices: List[str], db_path: str, 
                          method: str = "parallel", n_workers: Optional[int] = None) -> H3SQLiteManager:
    """
    Complete workflow: Process H3 indices and save to SQLite database.
    
    Args:
        h3_indices: List of H3 cell indices to process
        db_path: Path for SQLite database
        method: Processing method ("parallel", "sequential", "generator")
        n_workers: Number of workers for parallel processing
        
    Returns:
        H3SQLiteManager instance (remember to close it!)
    """
    # Import your processing functions here
    from your_h3_module import process_h3_indices, h3_to_objects_parallel_generator
    
    # Initialize SQLite manager
    db_manager = H3SQLiteManager(db_path, batch_size=500)
    
    print(f"Processing {len(h3_indices)} H3 indices using {method} method...")
    
    if method == "batch_parallel":
        # Process all at once, then save as batch
        results = process_h3_indices(h3_indices, method="parallel", n_workers=n_workers)
        db_manager.add_batch(results)
    
    elif method == "streaming":
        # Process and save one by one (most memory efficient)
        for h3_index, result in h3_to_objects_parallel_generator(h3_indices, n_workers):
            db_manager.add_result(h3_index, result)
    
    else:
        # Default: process all then save
        results = process_h3_indices(h3_indices, method=method, n_workers=n_workers)
        db_manager.add_batch(results)
    
    db_manager.finalize()
    return db_manager


if __name__ == "__main__":
    # Example usage
    sample_h3_indices = [
        "8928308280fffff",
        "8928308280bffff", 
        "89283082807ffff",
        "89283082803ffff"
    ]
    
    print("=== H3 SQLite Storage Example ===")
    
    # Example 1: Context manager usage (recommended)
    print("\n1. Using context manager:")
    with H3SQLiteManager("./output/h3_data.db") as db:
        # Simulate adding results (replace with your actual processing)
        for h3_index in sample_h3_indices:
            result = {
                "h3_index": h3_index,
                "centroid": (37.7749, -122.4194),  # Example coordinates
                "boundary": [(37.77, -122.42), (37.78, -122.41)],  # Example boundary
                "latlng_coords": [[37.77, -122.42], [37.78, -122.41]],
                "polygon": Polygon([(37.77, -122.42), (37.78, -122.41), (37.77, -122.42)])
            }
            db.add_result(h3_index, result)
        
        # Query examples
        print("   Database statistics:", db.get_statistics())
        
        # Get specific cell
        cell_data = db.get_result(sample_h3_indices[0])
        if cell_data:
            print(f"   Retrieved: {cell_data['h3_index']} -> {cell_data['centroid']}")
    
    print("\n2. Usage patterns:")
    print("   # Save large dataset:")
    print("   with H3SQLiteManager('./output/h3_data.db') as db:")
    print("       for h3_index, result in h3_to_objects_parallel_generator(h3_list):")
    print("           db.add_result(h3_index, result)")
    
    print("\n   # Query later:")
    print("   with H3SQLiteManager('./output/h3_data.db') as db:")
    print("       cell = db.get_result('8928308280fffff')")
    print("       area_range = db.query_by_area_range(0.001, 0.01)")
    print("       stats = db.get_statistics()")

