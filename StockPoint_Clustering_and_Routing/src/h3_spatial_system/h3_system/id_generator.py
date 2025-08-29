"""
ID generation for H3-based address system.
"""

import h3
import base36
import math
from typing import Dict, Tuple, Optional
import logging

from config.settings import ID_CONFIG, COUNTRY_CONFIG

logger = logging.getLogger(__name__)


class IDGenerator:
    """Generates dual ID system for H3 cells."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.id_config = config or ID_CONFIG
        self.country_config = COUNTRY_CONFIG
        
    def generate_h3_derived_id(self, h3_cell_id: str, admin_hierarchy: Dict) -> str:
        """
        Generate H3-derived ID: NG-STATE-LGA-WARD-XXXX
        
        Args:
            h3_cell_id: H3 cell identifier
            admin_hierarchy: Administrative hierarchy dictionary
            
        Returns:
            H3-derived address ID
        """
        try:
            # Convert H3 string to integer
            h3_int = h3.str_to_int(h3_cell_id)
            
            # Convert to base36 and get suffix
            h3_base36 = base36.dumps(h3_int)
            cell_suffix = h3_base36[-self.id_config['h3_suffix_length']:].upper()
            
            # Build hierarchical ID
            country_code = self.country_config['code']
            state_code = admin_hierarchy.get('state', {}).get('code', 'XX')
            lga_code = admin_hierarchy.get('lga', {}).get('code', 'XX')
            ward_code = admin_hierarchy.get('ward', {}).get('code', 'XX')
            
            # Construct ID with separator
            separator = self.id_config['separator']
            h3_derived_id = f"{country_code}{separator}{state_code}{separator}{lga_code}{separator}{ward_code}{separator}{cell_suffix}"
            
            return h3_derived_id
            
        except Exception as e:
            logger.error(f"Failed to generate H3-derived ID for {h3_cell_id}: {e}")
            return f"NG-XX-XX-XX-XXXX"
    
    def generate_grid_id(self, h3_cell_id: str, admin_hierarchy: Dict, admin_bounds: Dict) -> str:
        """
        Generate geographic grid ID: NG-STATE-LGA-WARD-XXXYYY
        
        Args:
            h3_cell_id: H3 cell identifier
            admin_hierarchy: Administrative hierarchy dictionary
            admin_bounds: Administrative unit bounds for normalization
            
        Returns:
            Grid-based address ID
        """
        try:
            # Get H3 cell centroid
            lat, lng = h3.cell_to_latlng(h3_cell_id)
            
            # Normalize coordinates to grid within admin unit bounds
            x_pos = self._normalize_to_grid(lng, admin_bounds.get('lng_range', [0, 1]))
            y_pos = self._normalize_to_grid(lat, admin_bounds.get('lat_range', [0, 1]))
            
            # Format with specified precision
            precision = self.id_config['grid_precision']
            x_str = f"{x_pos:0{precision}d}"
            y_str = f"{y_pos:0{precision}d}"
            
            # Build hierarchical ID
            country_code = self.country_config['code']
            state_code = admin_hierarchy.get('state', {}).get('code', 'XX')
            lga_code = admin_hierarchy.get('lga', {}).get('code', 'XX')
            ward_code = admin_hierarchy.get('ward', {}).get('code', 'XX')
            
            # Construct ID
            separator = self.id_config['separator']
            grid_id = f"{country_code}{separator}{state_code}{separator}{lga_code}{separator}{ward_code}{separator}{x_str}{y_str}"
            
            return grid_id
            
        except Exception as e:
            logger.error(f"Failed to generate grid ID for {h3_cell_id}: {e}")
            return f"NG-XX-XX-XX-000000"
    
    def _normalize_to_grid(self, value: float, bounds: list) -> int:
        """
        Normalize a coordinate value to grid position within bounds.
        
        Args:
            value: Coordinate value (lat or lng)
            bounds: [min_value, max_value] for the administrative unit
            
        Returns:
            Grid position (0-999 for 3-digit precision)
        """
        min_val, max_val = bounds
        
        if max_val == min_val:
            return 500  # Default to middle if no range
            
        # Normalize to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        
        # Convert to grid position (0 to 10^precision - 1)
        max_grid = 10 ** self.id_config['grid_precision'] - 1
        grid_pos = int(normalized * max_grid)
        
        # Ensure within bounds
        return max(0, min(grid_pos, max_grid))
    
    def generate_dual_ids(self, h3_cell_id: str, admin_hierarchy: Dict, admin_bounds: Dict) -> Dict[str, str]:
        """
        Generate both H3-derived and grid IDs for a cell.
        
        Args:
            h3_cell_id: H3 cell identifier
            admin_hierarchy: Administrative hierarchy dictionary
            admin_bounds: Administrative unit bounds
            
        Returns:
            Dictionary with both ID types
        """
        h3_derived_id = self.generate_h3_derived_id(h3_cell_id, admin_hierarchy)
        grid_id = self.generate_grid_id(h3_cell_id, admin_hierarchy, admin_bounds)
        
        return {
            'h3_derived_id': h3_derived_id,
            'grid_position_id': grid_id,
            'primary_address_id': h3_derived_id  # Default to H3-derived as primary
        }
    
    def validate_id_format(self, address_id: str) -> bool:
        """
        Validate the format of an address ID.
        
        Args:
            address_id: Address ID to validate
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = address_id.split(self.id_config['separator'])
            
            # Check basic structure
            if len(parts) != 5:
                return False
                
            country_code, state_code, lga_code, ward_code, suffix = parts
            
            # Validate country code
            if country_code != self.country_config['code']:
                return False
                
            # Validate suffix length (either 7 for H3 or 6 for grid)
            if len(suffix) not in [7, 6]:
                return False
                
            # Validate codes are not empty
            if not all([state_code, lga_code, ward_code]):
                return False
                
            return True
            
        except Exception:
            return False
    
    def parse_address_id(self, address_id: str) -> Dict[str, str]:
        """
        Parse an address ID into its components.
        
        Args:
            address_id: Address ID to parse
            
        Returns:
            Dictionary with parsed components
        """
        if not self.validate_id_format(address_id):
            raise ValueError(f"Invalid address ID format: {address_id}")
            
        parts = address_id.split(self.config['separator'])
        country_code, state_code, lga_code, ward_code, suffix = parts
        
        return {
            'country_code': country_code,
            'state_code': state_code,
            'lga_code': lga_code,
            'ward_code': ward_code,
            'cell_suffix': suffix,
            'id_type': 'h3_derived' if len(suffix) == 7 else 'grid'
        }


def generate_sample_ids() -> Dict[str, str]:
    """Generate sample IDs for testing."""
    generator = IDGenerator()
    
    # Sample data
    h3_cell_id = "8c1234567890abc"
    admin_hierarchy = {
        'state': {'code': 'LA', 'name': 'Lagos'},
        'lga': {'code': 'IK', 'name': 'Ikeja'},
        'ward': {'code': 'WA', 'name': 'Ward A'}
    }
    admin_bounds = {
        'lat_range': [6.0, 7.0],
        'lng_range': [3.0, 4.0]
    }
    
    return generator.generate_dual_ids(h3_cell_id, admin_hierarchy, admin_bounds)


if __name__ == "__main__":
    # Test ID generation
    sample_ids = generate_sample_ids()
    print("Sample IDs:")
    for id_type, id_value in sample_ids.items():
        print(f"  {id_type}: {id_value}")
    
    # Test validation
    generator = IDGenerator()
    test_id = sample_ids['h3_derived_id']
    is_valid = generator.validate_id_format(test_id)
    print(f"\nValidation test: {test_id} -> {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Test parsing
    parsed = generator.parse_address_id(test_id)
    print(f"Parsed components: {parsed}") 