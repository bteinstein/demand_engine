"""
Administrative assignment for H3 cells based on spatial overlap.
"""

import h3
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from .utils import h3_to_shapely_polygon

from config.settings import COVERAGE_THRESHOLD, QUALITY_CONFIG

# Set logging level for geometry warnings
import logging
logging.getLogger('src.h3_system.admin_assignment').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


from shapely.geometry import mapping

class AdministrativeAssignment:
    """Assigns H3 cells to administrative units based on spatial overlap."""
    
    def __init__(self, coverage_threshold: Optional[float] = None, h3_resolution: int = 8):
        self.coverage_threshold = coverage_threshold or COVERAGE_THRESHOLD
        self.h3_resolution = h3_resolution
        self.admin_units = {}
        self.h3_to_admin_map = {}
        self.boundary_cells = set()
        
        # Performance optimizations
        self._spatial_index = {}  # R-tree spatial index for each level
        self._assignment_cache = {}  # Cache for H3 cell assignments
        self._cache_hits = 0
        self._cache_misses = 0
        
    def load_admin_boundaries(self, states_path: str, lgas_path: str, wards_path: str):
        """Load administrative boundary data."""
        try:
            logger.info("Loading administrative boundaries...")
            
            # Load states
            states_gdf = gpd.read_file(states_path)
            self.admin_units['states'] = self._prepare_admin_data(states_gdf, 'state')
            
            # Load LGAs
            lgas_gdf = gpd.read_file(lgas_path)
            self.admin_units['lgas'] = self._prepare_admin_data(lgas_gdf, 'lga')
            
            # Load wards
            wards_gdf = gpd.read_file(wards_path)
            self.admin_units['wards'] = self._prepare_admin_data(wards_gdf, 'ward')
            
            logger.info(f"Loaded {len(self.admin_units['states'])} states, "
                       f"{len(self.admin_units['lgas'])} LGAs, "
                       f"{len(self.admin_units['wards'])} wards")
            
            # Validate loaded data
            self._validate_loaded_data()
            
            # Build spatial indexes for performance
            self._build_spatial_indexes()

            # Precompute H3 coverage for all wards
            # self.precompute_h3_coverage()
            
        except Exception as e:
            logger.error(f"Failed to load administrative boundaries: {e}")
            raise
    
    def _prepare_admin_data(self, gdf: gpd.GeoDataFrame, level: str) -> List[Dict]:
        """Prepare administrative data for efficient processing."""
        admin_units = []
        invalid_count = 0
        
        for _, row in gdf.iterrows():
            # Validate geometry before processing
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                invalid_count += 1
                logger.debug(f"Skipping null/empty geometry for {level} {row.get('name', 'Unknown')}")
                continue
            
            # Try to repair invalid geometries
            if not geometry.is_valid:
                try:
                    geometry = make_valid(geometry)
                    if geometry is None or geometry.is_empty:
                        invalid_count += 1
                        logger.debug(f"Could not repair geometry for {level} {row.get('name', 'Unknown')}")
                        continue
                except Exception:
                    invalid_count += 1
                    logger.debug(f"Failed to repair geometry for {level} {row.get('name', 'Unknown')}")
                    continue
            
            unit = {
                'geometry': geometry,
                'level': level
            }
            
            # Extract name and code based on level (using standardized column names)
            if level == 'state':
                unit['name'] = row.get('state_name', row.get('name', 'Unknown'))
                unit['code'] = row.get('state_code', row.get('code', 'XX'))
            elif level == 'lga':
                unit['name'] = row.get('lga_name', row.get('name', 'Unknown'))
                unit['code'] = row.get('lga_code', row.get('code', 'XX'))
                unit['state_name'] = row.get('state_name', 'Unknown')
                unit['state_code'] = row.get('state_code', 'XX')
            elif level == 'ward':
                unit['name'] = row.get('ward_name', row.get('name', 'Unknown'))
                unit['code'] = row.get('ward_code', row.get('code', 'XX'))
                unit['lga_name'] = row.get('lga_name', 'Unknown')
                unit['lga_code'] = row.get('lga_code', 'XX')
                unit['state_name'] = row.get('state_name', 'Unknown')
                unit['state_code'] = row.get('state_code', 'XX')
            
            admin_units.append(unit)
        
        if invalid_count > 0:
            logger.info(f"Filtered out {invalid_count} invalid geometries from {level} data")
        
        # Debug: Check for any remaining None geometries
        none_count = sum(1 for unit in admin_units if unit.get('geometry') is None)
        if none_count > 0:
            logger.warning(f"Found {none_count} units with None geometry in {level} data after filtering")
        
        return admin_units
    
    def _validate_loaded_data(self):
        """Validate that all loaded admin units have valid geometries."""
        for level, units in self.admin_units.items():
            none_count = 0
            invalid_count = 0
            
            for unit in units:
                geometry = unit.get('geometry')
                if geometry is None:
                    none_count += 1
                elif not hasattr(geometry, 'is_valid') or not geometry.is_valid:
                    invalid_count += 1
            
            if none_count > 0 or invalid_count > 0:
                logger.warning(f"{level}: {none_count} None geometries, {invalid_count} invalid geometries")
            else:
                logger.info(f"{level}: All {len(units)} units have valid geometries")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self._assignment_cache)
        }
    
    def _build_spatial_indexes(self):
        """Build spatial indexes for efficient spatial queries."""
        try:
            from rtree import index
            
            for level, units in self.admin_units.items():
                # Create R-tree spatial index
                idx = index.Index()
                
                for i, unit in enumerate(units):
                    geometry = unit.get('geometry')
                    if geometry is not None and geometry.is_valid:
                        # Get bounding box
                        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
                        idx.insert(i, bounds)
                
                self._spatial_index[level] = idx
                logger.info(f"Built spatial index for {level}: {len(units)} units")
                
        except ImportError:
            logger.warning("Rtree not available, spatial indexing disabled")
            self._spatial_index = {}
        except Exception as e:
            logger.warning(f"Failed to build spatial indexes: {e}")
            self._spatial_index = {}

    def precompute_h3_coverage(self):
        """Precompute H3 cell coverage for all administrative units."""
        logger.info("Precomputing H3 coverage for all wards...")
        
        # Create a lookup for LGAs and states by code for quick access
        lgas_by_code = {lga['code']: lga for lga in self.admin_units['lgas']}
        states_by_code = {state['code']: state for state in self.admin_units['states']}

        for ward in tqdm(self.admin_units['wards'], desc="Precomputing ward coverage"):
            try:
                # Get the geometry of the ward
                ward_geom = ward['geometry']
                if not ward_geom or not ward_geom.is_valid:
                    continue

                # Convert the ward geometry to the format H3 expects
                geo_json = gpd.GeoSeries([ward_geom]).__geo_interface__
                
                # Get all H3 cells that are covered by this ward
                h3_cells = h3.polygon_to_cells(geo_json, self.h3_resolution)

                # Get parent LGA and state
                lga = lgas_by_code.get(ward['lga_code'])
                state = states_by_code.get(ward['state_code'])

                for cell in h3_cells:
                    if cell not in self.h3_to_admin_map:
                        # This is the first time we've seen this cell, so assign it
                        self.h3_to_admin_map[cell] = {
                            'ward': ward,
                            'lga': lga,
                            'state': state
                        }
                    else:
                        # This cell is in more than one ward, mark it as a boundary cell
                        self.boundary_cells.add(cell)

            except Exception as e:
                logger.warning(f"Could not process ward {ward.get('name', 'Unknown')}: {e}")

        logger.info(f"Precomputation complete. Found {len(self.boundary_cells)} boundary cells.")
    
    def _assign_h3_cell_by_coverage(self, h3_cell_id: str) -> Dict:
        """
        Assign a single H3 cell to administrative units with caching.
        
        Args:
            h3_cell_id: H3 cell identifier
            
        Returns:
            Assignment result with administrative hierarchy and quality metrics
        """
        # Check cache first
        if h3_cell_id in self._assignment_cache:
            self._cache_hits += 1
            return self._assignment_cache[h3_cell_id]
        
        self._cache_misses += 1
        
        try:
            # Get H3 cell geometry
            # h3_boundary = h3.cell_to_boundary(h3_cell_id)
            # h3_polygon = Polygon(h3_boundary)
            # 
            h3_polygon = h3_to_shapely_polygon(h3_cell_id)
            
            # Get centroid for point-based assignment
            lat, lng = h3.cell_to_latlng(h3_cell_id)
            h3_point = Point(lng, lat)
            
            # Assign to each administrative level
            state_assignment = self._assign_to_level(h3_polygon, h3_point, 'states')
            lga_assignment = self._assign_to_level(h3_polygon, h3_point, 'lgas', 
                                                 parent_filter=state_assignment)
            ward_assignment = self._assign_to_level(h3_polygon, h3_point, 'wards',
                                                  parent_filter=lga_assignment)
            
            # Build administrative hierarchy
            admin_hierarchy = {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': state_assignment['assigned_unit'],
                'lga': lga_assignment['assigned_unit'],
                'ward': ward_assignment['assigned_unit']
            }
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                state_assignment, lga_assignment, ward_assignment
            )
            
            result = {
                'h3_id': h3_cell_id,
                'admin_assignment': admin_hierarchy,
                'assignment_quality': quality_metrics,
                'geometry': {
                    'centroid': {'lat': lat, 'lng': lng},
                    'area_km2': h3.cell_area(h3_cell_id, unit='km^2')
                }
            }
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._assignment_cache) < 10000:
                self._assignment_cache[h3_cell_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to assign H3 cell {h3_cell_id}: {e}")
            return self._create_fallback_assignment(h3_cell_id)

    def _build_assignment_result(self, h3_cell_id: str, state: Dict, lga: Dict, ward: Dict) -> Dict:
        """Build the assignment result dictionary."""
        lat, lng = h3.cell_to_latlng(h3_cell_id)
        admin_hierarchy = {
            'country': {'code': 'NG', 'name': 'Nigeria'},
            'state': state,
            'lga': lga,
            'ward': ward
        }
        quality_metrics = {
            'confidence_level': 'confident',
            'coverage_percentage': 100.0,
            'state_coverage': 100.0,
            'lga_coverage': 100.0,
            'ward_coverage': 100.0,
            'total_candidates_checked': 0
        }
        return {
            'h3_id': h3_cell_id,
            'admin_assignment': admin_hierarchy,
            'assignment_quality': quality_metrics,
            'geometry': {
                'centroid': {'lat': lat, 'lng': lng},
                'area_km2': h3.cell_area(h3_cell_id, unit='km^2')
            }
        }

    def _assign_h3_cell_by_coverage_(self, h3_cell_id: str) -> Dict:
        # Check cache first
        if h3_cell_id in self._assignment_cache:
            self._cache_hits += 1
            return self._assignment_cache[h3_cell_id]
        
        self._cache_misses += 1
        
        try:
            # Get H3 cell geometry
            h3_boundary = h3.cell_to_boundary(h3_cell_id)
            h3_polygon = Polygon(h3_boundary)
            
            # Get centroid for point-based assignment
            lat, lng = h3.cell_to_latlng(h3_cell_id)
            h3_point = Point(lng, lat)
            
            # Assign to each administrative level
            state_assignment = self._assign_to_level(h3_polygon, h3_point, 'states')
            lga_assignment = self._assign_to_level(h3_polygon, h3_point, 'lgas', 
                                                 parent_filter=state_assignment)
            ward_assignment = self._assign_to_level(h3_polygon, h3_point, 'wards',
                                                  parent_filter=lga_assignment)
            
            # Build administrative hierarchy
            admin_hierarchy = {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': state_assignment['assigned_unit'],
                'lga': lga_assignment['assigned_unit'],
                'ward': ward_assignment['assigned_unit']
            }
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                state_assignment, lga_assignment, ward_assignment
            )
            
            result = {
                'h3_id': h3_cell_id,
                'admin_assignment': admin_hierarchy,
                'assignment_quality': quality_metrics,
                'geometry': {
                    'centroid': {'lat': lat, 'lng': lng},
                    'area_km2': h3.cell_area(h3_cell_id, unit='km^2')
                }
            }
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._assignment_cache) < 10000:
                self._assignment_cache[h3_cell_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to assign H3 cell {h3_cell_id}: {e}")
            return self._create_fallback_assignment(h3_cell_id)
    
    def _assign_to_level(self, h3_polygon: Polygon, h3_point: Point, 
                        level: str, parent_filter: Optional[Dict] = None) -> Dict:
        """
        Assign H3 cell to a specific administrative level.
        
        Args:
            h3_polygon: H3 cell polygon
            h3_point: H3 cell centroid point
            level: Administrative level ('states', 'lgas', 'wards')
            parent_filter: Filter for parent administrative unit
            
        Returns:
            Assignment result for this level
        """
        all_candidates = self.admin_units[level]
        
        # Use spatial indexing if available
        if level in self._spatial_index and self._spatial_index[level] is not None:
            # Get H3 cell bounding box
            h3_bounds = h3_polygon.bounds  # (minx, miny, maxx, maxy)
            
            # Query spatial index for potential candidates
            candidate_indices = list(self._spatial_index[level].intersection(h3_bounds))
            candidates = [all_candidates[i] for i in candidate_indices]
        else:
            candidates = all_candidates
        
        # Filter by parent if specified
        if parent_filter:
            parent_code = parent_filter['assigned_unit']['code']
            if level == 'lgas':
                candidates = [c for c in candidates if c.get('state_code') == parent_code]
            elif level == 'wards':
                candidates = [c for c in candidates if c.get('lga_code') == parent_code]
        
        if not candidates:
            return self._create_fallback_level_assignment(level)
        
        # Calculate overlaps
        overlaps = []
        for unit in candidates:
            try:
                # Validate geometry before processing
                geometry = unit.get('geometry')
                if geometry is None:
                    continue
                
                # Additional safety checks
                if not hasattr(geometry, 'is_valid') or not hasattr(geometry, 'is_empty'):
                    continue
                
                if not geometry.is_valid or geometry.is_empty:
                    continue
                
                # Calculate intersection
                intersection = h3_polygon.intersection(geometry)
                if intersection.is_empty:
                    continue
                    
                overlap_area = intersection.area
                coverage_percentage = (overlap_area / h3_polygon.area) * 100
                
                overlaps.append({
                    'unit': unit,
                    'overlap_area': overlap_area,
                    'coverage_percentage': coverage_percentage
                })
                
            except Exception as e:
                # Only log if it's not a NoneType error (which we should have caught)
                if "'NoneType' object has no attribute" not in str(e):
                    logger.debug(f"Error calculating overlap for {unit.get('name', 'Unknown')}: {e}")
                continue
        
        if not overlaps:
            # Fallback to point-in-polygon check
            return self._point_based_assignment(h3_point, candidates, level)
        
        # Sort by coverage percentage
        overlaps.sort(key=lambda x: x['coverage_percentage'], reverse=True)
        best_overlap = overlaps[0]
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(
            best_overlap['coverage_percentage'], overlaps
        )
        
        return {
            'assigned_unit': best_overlap['unit'],
            'coverage_percentage': best_overlap['coverage_percentage'],
            'confidence_level': confidence_level, 
            'assignment_method': 'confident',
            'overlapping_units': overlaps[:3],  # Top 3 overlaps
            'total_candidates': len(candidates)
        }
    
    def _point_based_assignment(self, h3_point: Point, candidates: List[Dict], 
                           level: str, h3_polygon: Optional[Polygon] = None) -> Dict:
        """
        Enhanced fallback assignment using point-in-polygon check with distance-based scoring.
        Implements the 4-tier boundary case resolution approach.
        """
        containing_units = []
        distance_scores = []
        
        # First pass: Find units containing the point and calculate distances
        for unit in candidates:
            try:
                geometry = unit.get('geometry')
                if not self._is_valid_geometry(geometry):
                    continue
                
                is_contained = geometry.contains(h3_point)
                
                # Calculate distance to boundary (negative if inside, positive if outside)
                try:
                    distance_to_boundary = geometry.boundary.distance(h3_point)
                    if not is_contained:
                        distance_to_boundary = -distance_to_boundary  # Make negative for outside
                except Exception:
                    distance_to_boundary = float('inf') if not is_contained else 0
                
                unit_info = {
                    'unit': unit,
                    'contains_point': is_contained,
                    'distance_to_boundary': distance_to_boundary,
                    'boundary_distance_score': self._calculate_distance_score(distance_to_boundary)
                }
                
                if is_contained:
                    containing_units.append(unit_info)
                else:
                    distance_scores.append(unit_info)
                    
            except Exception as e:
                logger.debug(f"Error in point-based check for {unit.get('name', 'Unknown')}: {e}")
                continue
        
        # If point is contained in one or more units
        if containing_units:
            if len(containing_units) == 1:
                # Single container - use your existing confidence logic
                unit_info = containing_units[0]
                overlaps = [{'unit': unit_info['unit'], 'coverage_percentage': 100.0}]
                confidence_level = self._determine_confidence_level(100.0, overlaps)
                
                return {
                    'assigned_unit': unit_info['unit'],
                    'coverage_percentage': 100.0,
                    'confidence_level': confidence_level,
                    'assignment_method': 'point_contained',
                    'overlapping_units': overlaps,
                    'total_candidates': len(candidates)
                }
            else:
                # Multiple containers (overlapping boundaries) - choose closest to interior
                best_unit = max(containing_units, key=lambda x: x['distance_to_boundary'])
                # Reduce confidence due to boundary overlap
                coverage_pct = 85.0
                overlaps = [{'unit': u['unit'], 'coverage_percentage': coverage_pct - i*5} 
                        for i, u in enumerate(containing_units[:3])]
                confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
                
                return {
                    'assigned_unit': best_unit['unit'],
                    'coverage_percentage': coverage_pct,
                    'confidence_level': confidence_level,
                    'assignment_method': 'point_contained_multiple',
                    'overlapping_units': overlaps,
                    'total_candidates': len(candidates)
                }
        
        # Point not contained - apply boundary case resolution
        if not distance_scores:
            return self._create_fallback_level_assignment(level)
        
        # Sort by proximity (closest boundary first)
        distance_scores.sort(key=lambda x: abs(x['distance_to_boundary']))
        
        # Apply 4-tier boundary case resolution
        return self._apply_boundary_case_resolution(
            h3_point, distance_scores, level, h3_polygon
        )

    def _apply_boundary_case_resolution(self, h3_point: Point, distance_scores: List[Dict], 
                                    level: str, h3_polygon: Optional[Polygon] = None) -> Dict:
        """
        Implement the 4-tier sequential approach for boundary cases.
        """
        closest_units = distance_scores[:3]  # Consider top 3 closest
        
        # Tier 1: Intelligent Auto-Assignment (60% of cases)
        auto_assignment = self._intelligent_auto_assignment(closest_units, h3_point)
        if auto_assignment:
            return auto_assignment
        
        # Tier 2: Neighbor Consensus (25% of cases)
        if h3_polygon:
            neighbor_assignment = self._neighbor_consensus_assignment(
                h3_polygon, closest_units, level
            )
            if neighbor_assignment:
                return neighbor_assignment
        
        # Tier 3 & 4: Stakeholder Review or Temporary Assignment
        return self._final_boundary_assignment(closest_units, level, h3_point)

    def _intelligent_auto_assignment(self, closest_units: List[Dict], h3_point: Point) -> Optional[Dict]:
        """
        Tier 1: Intelligent auto-assignment with geometric rules.
        """
        if not closest_units:
            return None
        
        closest = closest_units[0]
        closest_distance = abs(closest['distance_to_boundary'])
        
        # Rule 1: If significantly closer to one boundary (>2x closer)
        if len(closest_units) > 1:
            second_closest_distance = abs(closest_units[1]['distance_to_boundary'])
            if closest_distance * 2 < second_closest_distance:
                coverage_pct = 35.0  # Just below 40% threshold
                overlaps = [{'unit': u['unit'], 'coverage_percentage': coverage_pct - i*5} 
                        for i, u in enumerate(closest_units[:3])]
                confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
                
                return {
                    'assigned_unit': closest['unit'],
                    'coverage_percentage': coverage_pct,
                    'confidence_level': confidence_level,
                    'assignment_method': 'auto_assignment_distance',
                    'resolution_tier': 1,
                    'overlapping_units': overlaps,
                    'total_candidates': len(closest_units)
                }
        
        # Rule 2: Very close to boundary (< threshold distance)
        if closest_distance < self._get_close_boundary_threshold():
            coverage_pct = 30.0
            overlaps = [{'unit': u['unit'], 'coverage_percentage': coverage_pct - i*3} 
                    for i, u in enumerate(closest_units[:3])]
            confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
            
            return {
                'assigned_unit': closest['unit'],
                'coverage_percentage': coverage_pct,
                'confidence_level': confidence_level,
                'assignment_method': 'auto_assignment_proximity',
                'resolution_tier': 1,
                'overlapping_units': overlaps,
                'total_candidates': len(closest_units)
            }
        
        return None

    def _neighbor_consensus_assignment(self, h3_polygon: Polygon, closest_units: List[Dict], 
                                    level: str) -> Optional[Dict]:
        """
        Tier 2: Analyze surrounding H3 cells for consensus.
        """
        try:
            # Get neighboring H3 cells (you'll need to implement this based on your H3 setup)
            neighbor_assignments = self._get_neighbor_assignments(h3_polygon, level)
            
            if not neighbor_assignments:
                return None
            
            # Count assignments by administrative unit
            unit_counts = {}
            for assignment in neighbor_assignments:
                unit_code = assignment.get('assigned_unit', {}).get('code')
                if unit_code:
                    unit_counts[unit_code] = unit_counts.get(unit_code, 0) + 1
            
            total_neighbors = len(neighbor_assignments)
            if total_neighbors < 3:  # Need sufficient neighbors for consensus
                return None
            
            # Check for 70%+ consensus
            for unit_code, count in unit_counts.items():
                consensus_percentage = (count / total_neighbors) * 100
                if consensus_percentage >= 70:
                    # Find the corresponding unit in closest_units
                    matching_unit = next(
                        (u for u in closest_units if u['unit'].get('code') == unit_code), 
                        None
                    )
                    if matching_unit:
                        coverage_pct = 25.0
                        overlaps = [{'unit': matching_unit['unit'], 'coverage_percentage': coverage_pct}]
                        confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
                        
                        return {
                            'assigned_unit': matching_unit['unit'],
                            'coverage_percentage': coverage_pct,
                            'confidence_level': confidence_level,
                            'assignment_method': 'neighbor_consensus',
                            'resolution_tier': 2,
                            'consensus_percentage': consensus_percentage,
                            'neighbor_sample_size': total_neighbors,
                            'overlapping_units': overlaps,
                            'total_candidates': len(closest_units)
                        }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in neighbor consensus: {e}")
            return None

    def _final_boundary_assignment(self, closest_units: List[Dict], level: str, 
                                h3_point: Point) -> Dict:
        """
        Tier 3 & 4: Stakeholder review or temporary assignment.
        """
        if not closest_units:
            return self._create_fallback_level_assignment(level)
        
        closest = closest_units[0]
        
        # Determine if this needs stakeholder review (Tier 3) or temporary assignment (Tier 4)
        needs_review = self._needs_stakeholder_review(h3_point, level)
        
        if needs_review:
            # Tier 3: Stakeholder Review
            coverage_pct = 20.0
            overlaps = [{'unit': u['unit'], 'coverage_percentage': coverage_pct - i*2} 
                    for i, u in enumerate(closest_units[:3])]
            confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
            
            return {
                'assigned_unit': closest['unit'],
                'coverage_percentage': coverage_pct,
                'confidence_level': confidence_level,
                'assignment_method': 'stakeholder_review_required',
                'resolution_tier': 3,
                'review_priority': 'high',
                'overlapping_units': overlaps,
                'total_candidates': len(closest_units)
            }
        else:
            # Tier 4: Temporary Assignment
            coverage_pct = 15.0
            overlaps = [{'unit': u['unit'], 'coverage_percentage': coverage_pct - i*2} 
                    for i, u in enumerate(closest_units[:3])]
            confidence_level = self._determine_confidence_level(coverage_pct, overlaps)
            
            return {
                'assigned_unit': closest['unit'],
                'coverage_percentage': coverage_pct,
                'confidence_level': confidence_level,
                'assignment_method': 'temporary_assignment',
                'resolution_tier': 4,
                'review_priority': 'low',
                'temporary_flag': True,
                'overlapping_units': overlaps,
                'total_candidates': len(closest_units)
            }

    def _calculate_distance_score(self, distance: float) -> float:
        """Calculate a score based on distance to boundary."""
        if distance == 0:
            return 100.0
        return max(0, 100 - abs(distance) * 10)  # Adjust multiplier based on your scale

    def _is_valid_geometry(self, geometry) -> bool:
        """Validate geometry object."""
        return (geometry is not None and 
                hasattr(geometry, 'is_valid') and 
                hasattr(geometry, 'is_empty') and 
                hasattr(geometry, 'contains') and
                geometry.is_valid and 
                not geometry.is_empty)

    def _get_close_boundary_threshold(self) -> float:
        """Get threshold for 'close to boundary' in your coordinate system units."""
        # Adjust this based on your coordinate system and precision needs
        return 0.001  # For geographic coordinates, roughly 100m

    def _needs_stakeholder_review(self, point: Point, level: str) -> bool:
        """
        Determine if a boundary case needs stakeholder review.
        High-priority areas: urban centers, infrastructure locations.
        """
        # Implement your logic here - this is a placeholder
        # You might check against urban area polygons, infrastructure databases, etc.
        return False  # Placeholder - adjust based on your criteria

    def _get_neighbor_assignments(self, h3_polygon: Polygon, level: str) -> List[Dict]:
        """
        Get assignments of neighboring H3 cells.
        You'll need to implement this based on your H3 and caching setup.
        """
        # Placeholder - implement based on your H3 neighbor logic
        # This might involve:
        # 1. Getting H3 cell ID from polygon
        # 2. Finding neighbor H3 cells
        # 3. Looking up their assignments (from cache or database)
        return []
        
    def _point_based_assignment_(self, h3_point: Point, candidates: List[Dict], 
                               level: str) -> Dict:
        """Fallback assignment using point-in-polygon check."""
        for unit in candidates:
            try:
                # Validate geometry before processing
                geometry = unit.get('geometry')
                if geometry is None:
                    continue
                
                # Additional safety checks
                if not hasattr(geometry, 'is_valid') or not hasattr(geometry, 'is_empty') or not hasattr(geometry, 'contains'):
                    continue
                
                if not geometry.is_valid or geometry.is_empty:
                    continue
                
                if geometry.contains(h3_point):
                    return {
                        'assigned_unit': unit,
                        'coverage_percentage': 100.0,  # Point is inside
                        'confidence_level': 'point_based',
                        'overlapping_units': [{'unit': unit, 'coverage_percentage': 100.0}],
                        'total_candidates': len(candidates)
                    }
            except Exception:
                continue
        
        # If no point-based assignment, return fallback
        return self._create_fallback_level_assignment(level)
    
    def _determine_confidence_level(self, coverage_percentage: float, 
                                   overlaps: List[Dict]) -> str:
        """Determine confidence level based on coverage and overlap distribution."""
        if coverage_percentage >= self.coverage_threshold:
            return 'confident'
        
        # Check if there's a clear winner
        if len(overlaps) > 1:
            second_best = overlaps[1]['coverage_percentage']
            if coverage_percentage > second_best * 2:  # Clear winner
                return 'confident'
        
        if coverage_percentage >= self.coverage_threshold * 0.75:  # 30% for 40% threshold
            return 'boundary_case'
        
        return 'manual_review'
    
    def _create_fallback_level_assignment(self, level: str) -> Dict:
        """Create fallback assignment for a specific level."""
        return {
            'assigned_unit': {
                'code': 'XX',
                'name': 'Unknown',
                'level': level
            },
            'coverage_percentage': 0.0,
            'confidence_level': 'manual_review',
            'overlapping_units': [],
            'total_candidates': 0
        }

    def _calculate_quality_metrics(self, state_assignment: Dict, 
                                  lga_assignment: Dict, 
                                  ward_assignment: Dict) -> Dict:
        """Calculate overall quality metrics for the assignment."""
        # Use the lowest confidence level as overall confidence
        confidence_levels = [
            state_assignment['confidence_level'],
            lga_assignment['confidence_level'],
            ward_assignment['confidence_level']
        ]
        
        overall_confidence = min(confidence_levels, 
                               key=lambda x: QUALITY_CONFIG['confidence_levels'].index(x))
        
        # Calculate average coverage
        coverage_percentages = [
            state_assignment['coverage_percentage'],
            lga_assignment['coverage_percentage'],
            ward_assignment['coverage_percentage']
        ]
        avg_coverage = np.mean(coverage_percentages)
        
        return {
            'confidence_level': overall_confidence,
            'coverage_percentage': avg_coverage,
            'state_coverage': state_assignment['coverage_percentage'],
            'lga_coverage': lga_assignment['coverage_percentage'],
            'ward_coverage': ward_assignment['coverage_percentage'],
            'total_candidates_checked': (
                state_assignment['total_candidates'] +
                lga_assignment['total_candidates'] +
                ward_assignment['total_candidates']
            )
        }
    
    def _create_fallback_assignment(self, h3_cell_id: str) -> Dict:
        """Create fallback assignment when assignment fails."""
        return {
            'h3_id': h3_cell_id,
            'admin_assignment': {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': {'code': 'XX', 'name': 'Unknown'},
                'lga': {'code': 'XX', 'name': 'Unknown'},
                'ward': {'code': 'XX', 'name': 'Unknown'}
            },
            'assignment_quality': {
                'confidence_level': 'manual_review',
                'coverage_percentage': 0.0,
                'error': 'Assignment failed'
            },
            'geometry': {
                'centroid': {'lat': 0.0, 'lng': 0.0},
                'area_km2': 0.0
            }
        }
       
    def assign_multiple_cells(self, h3_cell_ids: List[str]) -> List[Dict]:
        """
        Assign multiple H3 cells to administrative units with batch optimization.
        
        Args:
            h3_cell_ids: List of H3 cell identifiers
            
        Returns:
            List of assignment results
        """
        results = []
        
        # Process in batches for better performance
        batch_size = 1000
        for i in range(0, len(h3_cell_ids), batch_size):
            batch = h3_cell_ids[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for h3_id in batch:
                result = self.assign_h3_cell(h3_id)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Log progress
            if len(h3_cell_ids) > batch_size:
                logger.info(f"Processed {min(i + batch_size, len(h3_cell_ids))}/{len(h3_cell_ids)} cells")
        
        return results
    
    def get_assignment_statistics(self, assignments: List[Dict]) -> Dict:
        """Generate statistics for a set of assignments."""
        if not assignments:
            return {}
        
        confidence_counts = {}
        coverage_percentages = []
        total_cells = len(assignments)
        
        for assignment in assignments:
            quality = assignment['assignment_quality']
            confidence = quality['confidence_level']
            coverage = quality['coverage_percentage']
            
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
            coverage_percentages.append(coverage)
        
        return {
            'total_cells': total_cells,
            'confidence_distribution': confidence_counts,
            'avg_coverage_percentage': np.mean(coverage_percentages),
            'median_coverage_percentage': np.median(coverage_percentages),
            'min_coverage_percentage': np.min(coverage_percentages),
            'max_coverage_percentage': np.max(coverage_percentages),
            'confident_assignments': confidence_counts.get('confident', 0),
            'boundary_cases': confidence_counts.get('boundary_case', 0),
            'manual_review_needed': confidence_counts.get('manual_review', 0)
        }

    # Parallel Processing
    def _generate_addresses_parallel(self, h3_cells: List[str]) -> List[Dict]:
        """Generate addresses using parallel processing for larger datasets."""
        logger.info("Using parallel processing for address generation...")
        
        # Split cells into chunks for parallel processing
        chunk_size = self.processing_config['chunk_size']
        num_workers = self.processing_config.get('num_workers', os.cpu_count())
        
        # Create chunks
        chunks = [h3_cells[i:i + chunk_size] for i in range(0, len(h3_cells), chunk_size)]
        
        all_addresses = []
        
        # Initialize global instances for multiprocessing
        global admin_assignment_instance, id_generator_instance
        admin_assignment_instance = self.admin_assignment
        id_generator_instance = self.id_generator
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all chunks
                future_to_chunk = {
                    executor.submit(_process_chunk_worker, chunk): chunk 
                    for chunk in chunks
                }
                
                # Collect results with progress bar
                with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                    for future in as_completed(future_to_chunk):
                        try:
                            chunk_addresses = future.result()
                            all_addresses.extend(chunk_addresses)
                            pbar.update(1)
                        except Exception as e:
                            chunk = future_to_chunk[future]
                            logger.error(f"Failed to process chunk of {len(chunk)} cells: {e}")
                            # Add fallback records for failed chunk
                            for h3_id in chunk:
                                all_addresses.append(self._create_fallback_record(h3_id))
                            pbar.update(1)
                            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            return self._generate_addresses_sequential(h3_cells)
        
        self.address_data = all_addresses
        logger.info(f"Generated {len(all_addresses)} address records using parallel processing")
        
        return all_addresses


    def _process_chunk_worker(h3_cells: List[str]) -> List[Dict]:
        """
        Worker function for processing H3 cell chunks in parallel.
        This function runs in a separate process.
        """
        global admin_assignment_instance, id_generator_instance
        
        chunk_addresses = []
        
        for h3_id in h3_cells:
            try:
                # Step 1: Administrative assignment
                assignment_result = admin_assignment_instance._assign_h3_cell_by_coverage(h3_id)
                
                # Step 2: Generate IDs
                admin_hierarchy = assignment_result['admin_assignment']
                admin_bounds = _get_admin_bounds_worker(admin_hierarchy)
                ids = id_generator_instance.generate_dual_ids(h3_id, admin_hierarchy, admin_bounds)
                
                # Step 3: Combine results
                address_record = {
                    'h3_id': h3_id,
                    **ids,
                    'admin_assignment': admin_hierarchy,
                    'assignment_quality': assignment_result['assignment_quality'],
                    'geometry': assignment_result['geometry']
                }
                
                chunk_addresses.append(address_record)
                
            except Exception as e:
                logger.error(f"Failed to process H3 cell {h3_id}: {e}")
                # Add fallback record
                chunk_addresses.append(_create_fallback_record_worker(h3_id))
        
        return chunk_addresses


    def _get_admin_bounds_worker(admin_hierarchy: Dict) -> Dict:
        """Worker version of _get_admin_bounds method."""
        # Use Nigeria bounds as fallback
        # In full implementation, calculate bounds for specific admin unit
        return {
            'lat_range': [4.0, 14.0],  # Nigeria approximate bounds
            'lng_range': [2.5, 15.0]
        }


    def _create_fallback_record_worker(h3_id: str) -> Dict:
        """Worker version of _create_fallback_record method."""
        try:
            lat, lng = h3.cell_to_latlng(h3_id)
            area_km2 = h3.cell_area(h3_id, unit='km^2')
        except:
            lat, lng, area_km2 = 0.0, 0.0, 0.0
        
        return {
            'h3_id': h3_id,
            'h3_derived_id': 'NG-XX-XX-XX-XXXX',
            'grid_position_id': 'NG-XX-XX-XX-000000',
            'primary_address_id': 'NG-XX-XX-XX-XXXX',
            'admin_assignment': {
                'country': {'code': 'NG', 'name': 'Nigeria'},
                'state': {'code': 'XX', 'name': 'Unknown'},
                'lga': {'code': 'XX', 'name': 'Unknown'},
                'ward': {'code': 'XX', 'name': 'Unknown'}
            },
            'assignment_quality': {
                'confidence_level': 'manual_review',
                'coverage_percentage': 0.0,
                'error': 'Processing failed'
            },
            'geometry': {
                'centroid': {'lat': lat, 'lng': lng},
                'area_km2': area_km2
            }
        }

        

def test_assignment():
    """Test the administrative assignment functionality."""
    # Create a simple test case
    assignment = AdministrativeAssignment()
    
    # Mock administrative units for testing
    assignment.admin_units = {
        'states': [
            {
                'name': 'Lagos',
                'code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (4.0, 6.0), (4.0, 7.0), (3.0, 7.0)]),
                'level': 'state'
            }
        ],
        'lgas': [
            {
                'name': 'Ikeja',
                'code': 'IK',
                'state_code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (3.5, 6.0), (3.5, 6.5), (3.0, 6.5)]),
                'level': 'lga'
            }
        ],
        'wards': [
            {
                'name': 'Ward A',
                'code': 'WA',
                'lga_code': 'IK',
                'state_code': 'LA',
                'geometry': Polygon([(3.0, 6.0), (3.25, 6.0), (3.25, 6.25), (3.0, 6.25)]),
                'level': 'ward'
            }
        ]
    }
    
    # Test with a sample H3 cell
    test_h3_id = "8c1234567890abc"
    result = assignment.assign_h3_cell(test_h3_id)
    
    print("Test Assignment Result:")
    print(f"H3 ID: {result['h3_id']}")
    print(f"State: {result['admin_assignment']['state']['name']}")
    print(f"LGA: {result['admin_assignment']['lga']['name']}")
    print(f"Ward: {result['admin_assignment']['ward']['name']}")
    print(f"Confidence: {result['assignment_quality']['confidence_level']}")
    print(f"Coverage: {result['assignment_quality']['coverage_percentage']:.1f}%")


if __name__ == "__main__":
    test_assignment() 