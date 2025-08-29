"""
Data downloader for administrative boundary data from GRID3.
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import time
from tqdm import tqdm

from config.settings import ADMIN_DATA_SOURCES, RAW_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDownloader:
    """Downloads administrative boundary data from GRID3 sources."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from URL with progress tracking."""
        output_path = self.output_dir / filename
        
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path
            
        logger.info(f"Downloading {filename} from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
                        
            logger.info(f"Successfully downloaded: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            raise
    
    def download_admin_boundaries(self) -> Dict[str, Path]:
        """Download all administrative boundary datasets."""
        downloaded_files = {}
        
        for level, config in ADMIN_DATA_SOURCES.items():
            logger.info(f"Downloading {level} boundaries...")
            
            try:
                # For GRID3, we need to construct the actual download URL
                # The about page URL needs to be converted to a download URL
                download_url = self._get_download_url(config["url"])
                
                file_path = self.download_file(
                    download_url, 
                    config["filename"]
                )
                downloaded_files[level] = file_path
                
                # Add delay to be respectful to the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to download {level} boundaries: {e}")
                continue
                
        return downloaded_files
    
    def _get_download_url(self, about_url: str) -> str:
        """Convert GRID3 about page URL to actual download URL."""
        # GRID3 URLs typically follow this pattern:
        # about page: https://data.grid3.org/datasets/GRID3::dataset-name/about
        # download: https://data.grid3.org/datasets/GRID3::dataset-name/0/download
        
        if "about" in about_url:
            download_url = about_url.replace("/about", "/0/download")
            return download_url
        else:
            # If it's already a download URL, return as is
            return about_url
    
    def validate_downloaded_data(self, file_paths: Dict[str, Path]) -> Dict[str, bool]:
        """Validate downloaded files by checking if they can be loaded as GeoJSON."""
        validation_results = {}
        
        for level, file_path in file_paths.items():
            try:
                logger.info(f"Validating {level} data...")
                
                # Try to load as GeoJSON
                gdf = gpd.read_file(file_path)
                
                # Basic validation checks
                if gdf.empty:
                    raise ValueError(f"Empty dataset for {level}")
                
                if 'geometry' not in gdf.columns:
                    raise ValueError(f"No geometry column in {level}")
                
                # Rename columns to expected names and resave
                gdf = self._standardize_column_names(gdf, level)
                
                # Save the standardized version
                standardized_path = file_path.parent / f"{file_path.stem}_standardized{file_path.suffix}"
                gdf.to_file(standardized_path, driver='GeoJSON')
                logger.info(f"Saved standardized {level} data to: {standardized_path}")
                
                # Update the file path to use the standardized version
                file_paths[level] = standardized_path
                
                validation_results[level] = True
                logger.info(f"✓ {level} data validated and standardized successfully")
                
            except Exception as e:
                logger.error(f"✗ {level} data validation failed: {e}")
                validation_results[level] = False
                
        return validation_results
    
    def _standardize_column_names(self, gdf: gpd.GeoDataFrame, level: str) -> gpd.GeoDataFrame:
        """Rename columns to standardized names."""
        column_mapping = {}
        
        if level == 'states':
            column_mapping = {
                'statename': 'state_name',
                'statecode': 'state_code'
            }
        elif level == 'lgas':
            column_mapping = {
                'lganame': 'lga_name',
                'lgacode': 'lga_code',
                'statename': 'state_name',
                'statecode': 'state_code'
            }
        elif level == 'wards':
            column_mapping = {
                'wardname': 'ward_name',
                'wardcode': 'ward_code',
                'lganame': 'lga_name',
                'lgacode': 'lga_code',
                'statename': 'state_name',
                'statecode': 'state_code'
            }
        
        # Only rename columns that exist in the dataframe
        existing_mapping = {old: new for old, new in column_mapping.items() if old in gdf.columns}
        
        if existing_mapping:
            logger.info(f"Renaming columns for {level}: {existing_mapping}")
            gdf = gdf.rename(columns=existing_mapping)
        else:
            logger.info(f"No column renaming needed for {level}")
        
        return gdf
    
    def _get_required_columns(self, admin_level: str) -> list:
        """Get required columns for each administrative level."""
        base_columns = ['geometry']
        
        if admin_level == 'states':
            return base_columns + ['state_name', 'state_code']
        elif admin_level == 'lgas':
            return base_columns + ['lga_name', 'lga_code', 'state_name', 'state_code']
        elif admin_level == 'wards':
            return base_columns + ['ward_name', 'ward_code', 'lga_name', 'lga_code', 'state_name', 'state_code']
        else:
            return base_columns
    
    def get_data_summary(self, file_paths: Dict[str, Path]) -> Dict[str, Dict]:
        """Generate summary statistics for downloaded data."""
        summaries = {}
        
        for level, file_path in file_paths.items():
            try:
                gdf = gpd.read_file(file_path)
                
                summary = {
                    'total_features': len(gdf),
                    'columns': list(gdf.columns),
                    'geometry_type': str(gdf.geometry.geom_type.iloc[0]) if not gdf.empty else None,
                    'crs': str(gdf.crs),
                    'bounds': gdf.total_bounds.tolist() if not gdf.empty else None,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                
                # Add level-specific statistics (now using standardized column names)
                if level == 'states':
                    summary['unique_states'] = gdf['state_name'].nunique() if 'state_name' in gdf.columns else None
                elif level == 'lgas':
                    summary['unique_lgas'] = gdf['lga_name'].nunique() if 'lga_name' in gdf.columns else None
                    summary['states_covered'] = gdf['state_name'].nunique() if 'state_name' in gdf.columns else None
                elif level == 'wards':
                    summary['unique_wards'] = gdf['ward_name'].nunique() if 'ward_name' in gdf.columns else None
                    summary['lgas_covered'] = gdf['lga_name'].nunique() if 'lga_name' in gdf.columns else None
                
                summaries[level] = summary
                
            except Exception as e:
                logger.error(f"Failed to generate summary for {level}: {e}")
                summaries[level] = {'error': str(e)}
                
        return summaries


def download_admin_boundaries(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Convenience function to download all administrative boundaries."""
    downloader = DataDownloader(output_dir)
    return downloader.download_admin_boundaries()


def validate_and_summarize_data(file_paths: Dict[str, Path]) -> tuple:
    """Download, validate, and summarize administrative boundary data."""
    downloader = DataDownloader()
    
    # Validate data
    validation_results = downloader.validate_downloaded_data(file_paths)
    
    # Generate summaries
    summaries = downloader.get_data_summary(file_paths)
    
    return validation_results, summaries


if __name__ == "__main__":
    # Example usage
    downloader = DataDownloader()
    downloaded_files = downloader.download_admin_boundaries()
    
    if downloaded_files:
        validation_results, summaries = validate_and_summarize_data(downloaded_files)
        
        print("\n=== Validation Results ===")
        for level, is_valid in validation_results.items():
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"{level}: {status}")
        
        print("\n=== Data Summaries ===")
        for level, summary in summaries.items():
            print(f"\n{level.upper()}:")
            for key, value in summary.items():
                print(f"  {key}: {value}") 