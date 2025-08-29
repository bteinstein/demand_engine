"""
Configuration settings for the H3-based address system for Nigeria.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SQL_DIR = PROJECT_ROOT / '_sql'
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"

OUTPUT_DIR = PROJECT_ROOT / 'output'

# H3 Configuration
H3_RESOLUTION = 8  # ~0.74 km² per cell
COVERAGE_THRESHOLD = 40.0  # Minimum percentage for confident assignment

# Administrative boundary data sources
ADMIN_DATA_SOURCES = {
    "states": {
        "about-url": "https://data.grid3.org/datasets/GRID3::grid3-nga-operational-state-boundaries-/about",
        "url": "https://stg-arcgisazurecdataprod3.az.arcgis.com/exportfiles-61962-880/NGA_State_Boundaries_V2_-2781486175142980418.geojson?sv=2018-03-28&sr=b&sig=cFcJJy54XEQqkAQMDrgdaNSy4BpGJTNoetUJoNNJNqk%3D&se=2025-08-04T06%3A57%3A52Z&sp=r",
        "filename": "grid3-nga-operational-state-boundaries.geojson",
        "admin_level": 1,
        'standardize_file_path': RAW_DATA_DIR / 'grid3-nga-operational-state-boundaries_standardized.geojson'
    },
    "lgas": {
        "about-url": "https://data.grid3.org/datasets/GRID3::grid3-nga-operational-lga-boundaries/about",
        "url": "https://hub.arcgis.com/api/v3/datasets/2bb616a49ee84f409427cc2143787113_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1",
        "filename": "grid3-nga-operational-lga-boundaries.geojson",
        "admin_level": 2,
        'standardize_file_path': RAW_DATA_DIR / 'grid3-nga-operational-lga-boundaries_standardized.geojson'
    },
    "wards": {
        "about-url": "https://data.grid3.org/datasets/GRID3::grid3-nga-operational-wards-v1-0/about",
        "url": "https://hub.arcgis.com/api/v3/datasets/0824aded5f5a4d39b10871c667aa8ccf_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1",
        "filename": "grid3-nga-operational-wards-v1-0.geojson",
        "admin_level": 3,
        'standardize_file_path': RAW_DATA_DIR / 'grid3-nga-operational-wards-v1-0_standardized.geojson'
    }
}

INPUT_BASE_DATA_SOURCES = {
            'sp_dim': {
                'sql_path': SQL_DIR / 'sp_dim.sql',
                'local_file_path': RAW_DATA_DIR / 'df_sp_dim.parquet'
            },
            'sp_location_mapping': {
                'sql_path': SQL_DIR / 'sp_location_map.sql',
                'local_file_path': RAW_DATA_DIR / 'df_sp_location_mapping.parquet'
            },
            'customer_dim': {
                'sql_path': SQL_DIR / 'get_customer_dim.sql',
                'local_file_path': RAW_DATA_DIR / 'df_customer_dim.parquet'
            },
            'sp_active_customers': {
                'sql_path': SQL_DIR / 'sp_active_customers.sql',
                'local_file_path': RAW_DATA_DIR / 'df_sp_active_customers.parquet'
            },
        }


# Country configuration
COUNTRY_CONFIG = {
    "name": "Nigeria",
    "code": "NG",
    "bounds": {
        "min_lat": 4.0,
        "max_lat": 14.0,
        "min_lng": 2.5,
        "max_lng": 14.5
    }
}

# ID Generation Configuration
ID_CONFIG = {
    "h3_suffix_length": 7,
    "grid_precision": 3,  # Number of digits for x,y coordinates
    "separator": "-"
}

# Processing Configuration
PROCESSING_CONFIG = {
    "chunk_size": 10000,  # Number of H3 cells to process in chunks
    "max_workers": os.cpu_count(),     # Number of parallel workers
    "memory_limit_gb": 8  # Memory limit for processing
}

# Storage Configuration
STORAGE_CONFIG = {
    "default_format": "parquet",
    "compression": "snappy",
    "partition_by": ["state_code"],
    "duckdb_path": EXPORTS_DIR / "h3_addresses.duckdb",
    "h3_duckdb_path": EXPORTS_DIR / "h3_data.duckdb",
    "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
    "postgres_url": os.getenv("POSTGRES_URL", None)
}

# Quality Assurance
QUALITY_CONFIG = {
    "min_coverage_percentage": 40.0,
    "boundary_case_threshold": 10.0,  # Percentage of cells expected to be boundary cases
    "confidence_levels": ["confident", "boundary_case", "manual_review", "point_based"]
}

# Boundary Case Resolution
BOUNDARY_RESOLUTION_CONFIG = {
    "auto_assignment_threshold": 30.0,  # Minimum coverage for auto-assignment
    "neighbor_consensus_threshold": 70.0,  # Percentage of neighbors for consensus
    "stakeholder_review_priority": ["urban_centers", "infrastructure", "high_population"]
}

# Export Configuration
EXPORT_CONFIG = {
    "formats": ["parquet", "csv", "geojson"],
    "include_geometry": True,
    "include_metadata": True,
    "compression": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "h3_system.log"
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "target_lookup_time_ms": 100,
    "target_processing_time_seconds": 30,
    "memory_optimization": True,
    "use_spatial_index": True
}

# Validation Configuration
VALIDATION_CONFIG = {
    "validate_coverage": True,
    "validate_uniqueness": True,
    "validate_administrative_hierarchy": True,
    "generate_quality_report": True
} 