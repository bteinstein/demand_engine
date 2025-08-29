import pandas as pd
import geopandas as gpd
import h3
import numpy as np
from haversine import haversine, Unit
from shapely.geometry import Point
import networkx as nx
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def validate_input_data(sp_dim_df, customers_gdf, df_output_assignment):
    assert not sp_dim_df[['stock_point_id', 'latitude', 'longitude']].isnull().any().any()
    assert not df_output_assignment[['h3_cell_id', 'customer_id']].isnull().any().any()
    sample_h3 = df_output_assignment['h3_cell_id'].iloc[0]
    assert h3.get_resolution(sample_h3) == 8, "H3 cells must be resolution 8"
    assert sp_dim_df['latitude'].between(-90, 90).all()
    assert sp_dim_df['longitude'].between(-180, 180).all()
    print("✓ Data validation passed")

def extract_fulfillment_center(sp_dim_df, stock_point_id):
    fc_data = sp_dim_df[sp_dim_df['stock_point_id'] == stock_point_id]
    if fc_data.empty:
        raise ValueError(f"Stock point {stock_point_id} not found")
    fc_info = fc_data.iloc[0].to_dict()
    return {
        'stock_point_id': fc_info['stock_point_id'],
        'latitude': fc_info['latitude'],
        'longitude': fc_info['longitude'],
        'coordinates': (fc_info['latitude'], fc_info['longitude'])
    }

def calculate_h3_population_density(df_output_assignment):
    h3_population = (df_output_assignment
                    .groupby('h3_cell_id')['customer_id']
                    .nunique()
                    .reset_index())
    h3_population.columns = ['h3_cell_id', 'population_density']
    return h3_population

def haversine_distance(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.KILOMETERS)

def calculate_h3_centroid_distance(h3_cell_id, fc_lat, fc_lon):
    centroid_lat, centroid_lon = h3.cell_to_latlng(h3_cell_id)
    return haversine_distance(fc_lat, fc_lon, centroid_lat, centroid_lon)

def calculate_h3_farthest_vertex_distance(h3_cell_id, fc_lat, fc_lon):
    boundary = h3.cell_to_boundary(h3_cell_id)
    max_distance = 0
    for vertex_lat, vertex_lon in boundary:
        distance = haversine_distance(fc_lat, fc_lon, vertex_lat, vertex_lon)
        max_distance = max(max_distance, distance)
    return max_distance

def calculate_route_compactness(route_cells, fulfillment_center):
    if len(route_cells) <= 1:
        return 0
    
    fc_lat, fc_lon = fulfillment_center['latitude'], fulfillment_center['longitude']
    distances = []
    for cell_id in route_cells:
        centroid_lat, centroid_lon = h3.cell_to_latlng(cell_id)
        distance = haversine_distance(fc_lat, fc_lon, centroid_lat, centroid_lon)
        distances.append(distance)
    
    return np.std(distances) if len(distances) > 1 else 0

def build_h3_dataset(df_output_assignment, fulfillment_center):
    h3_population = calculate_h3_population_density(df_output_assignment)
    h3_metrics = []
    fc_lat, fc_lon = fulfillment_center['latitude'], fulfillment_center['longitude']
    
    for _, row in h3_population.iterrows():
        h3_cell_id = row['h3_cell_id']
        centroid_distance = calculate_h3_centroid_distance(h3_cell_id, fc_lat, fc_lon)
        vertex_distance = calculate_h3_farthest_vertex_distance(h3_cell_id, fc_lat, fc_lon)
        neighbors = list(h3.grid_disk(h3_cell_id, 1))
        neighbors.remove(h3_cell_id)
        
        h3_metrics.append({
            'h3_cell_id': h3_cell_id,
            'population_density': row['population_density'],
            'centroid_distance_km': centroid_distance,
            'vertex_distance_km': vertex_distance,
            'neighbors': neighbors,
            'centroid_coords': h3.cell_to_latlng(h3_cell_id)
        })
    
    return pd.DataFrame(h3_metrics)

def calculate_h3_score(population_density, distance, w1=1.0, w2=1.0, normalize=True):
    if normalize:
        pop_norm = population_density / 100
        dist_norm = distance / 10
        return w1 * pop_norm - w2 * dist_norm
    else:
        return w1 * population_density - w2 * distance

def add_scores_to_dataset(h3_dataset, w1=1.0, w2=1.0, compactness_weight=0.3):
    h3_dataset['score_centroid'] = h3_dataset.apply(
        lambda row: calculate_h3_score(row['population_density'], 
                                     row['centroid_distance_km'], w1, w2), axis=1)
    h3_dataset['score_vertex'] = h3_dataset.apply(
        lambda row: calculate_h3_score(row['population_density'], 
                                     row['vertex_distance_km'], w1, w2), axis=1)
    return h3_dataset

def build_adjacency_graph(h3_dataset):
    G = nx.Graph()
    h3_cells = set(h3_dataset['h3_cell_id'])
    
    for _, row in h3_dataset.iterrows():
        G.add_node(row['h3_cell_id'],
                  population_density=row['population_density'],
                  centroid_distance_km=row['centroid_distance_km'],
                  vertex_distance_km=row['vertex_distance_km'],
                  score_centroid=row['score_centroid'],
                  score_vertex=row['score_vertex'])
    
    for _, row in h3_dataset.iterrows():
        cell_id = row['h3_cell_id']
        for neighbor in row['neighbors']:
            if neighbor in h3_cells:
                G.add_edge(cell_id, neighbor)
    
    return G

class RouteBuilder:
    def __init__(self, seed_cell, route_id, fulfillment_center, distance_method):
        self.route_id = f"route_{route_id}"
        self.cells = [seed_cell['h3_cell_id']]
        self.total_customers = seed_cell['population_density']
        self.fulfillment_center = fulfillment_center
        self.distance_method = distance_method
        self.cell_data = {seed_cell['h3_cell_id']: seed_cell}

    def add_cell(self, cell_data):
        self.cells.append(cell_data['h3_cell_id'])
        self.total_customers += cell_data['population_density']
        self.cell_data[cell_data['h3_cell_id']] = cell_data

    def get_max_distance(self):
        distance_col = f'{self.distance_method}_distance_km'
        return max(self.cell_data[cell_id][distance_col] for cell_id in self.cells)

    def get_cumulative_distance(self):
        if len(self.cells) <= 1:
            return 0
        total_distance = 0
        for i in range(len(self.cells) - 1):
            coord1 = h3.cell_to_latlng(self.cells[i])
            coord2 = h3.cell_to_latlng(self.cells[i + 1])
            total_distance += haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
        return total_distance

    def get_cell_ids(self):
        return self.cells

    def get_compactness_penalty(self, compactness_weight):
        compactness = calculate_route_compactness(self.cells, self.fulfillment_center)
        return compactness_weight * compactness

def validate_enhanced_constraints(route, min_customers, min_distance, max_distance, 
                                max_clusters_per_route, max_customers_per_route, 
                                compactness_weight, max_compactness_penalty=2.0):
    distance = route.get_max_distance()
    customers = route.total_customers
    clusters = len(route.cells)
    compactness_penalty = route.get_compactness_penalty(compactness_weight)
    
    constraints = {
        'min_customers': customers >= min_customers,
        'min_distance': distance >= min_distance,
        'max_distance': distance <= max_distance,
        'max_clusters': clusters <= max_clusters_per_route,
        'max_customers': customers <= max_customers_per_route,
        'compactness': compactness_penalty <= max_compactness_penalty
    }
    
    return all(constraints.values())

def expand_route_greedily(route, h3_dataset, adjacency_graph, used_cells, 
                         min_customers, min_distance, max_distance, 
                         distance_method, score_method, 
                         max_clusters_per_route, max_customers_per_route, 
                         compactness_weight):
    while True:
        if len(route.cells) >= max_clusters_per_route:
            break
            
        if route.total_customers >= max_customers_per_route:
            break
            
        candidates = set()
        for cell_id in route.get_cell_ids():
            for neighbor in adjacency_graph.neighbors(cell_id):
                if neighbor not in used_cells:
                    candidates.add(neighbor)
        
        if not candidates:
            break
        
        candidate_scores = []
        for candidate in candidates:
            candidate_data = h3_dataset[h3_dataset['h3_cell_id'] == candidate].iloc[0]
            score = candidate_data[f'score_{score_method}']
            candidate_scores.append((score, candidate, candidate_data))
        
        candidate_scores.sort(reverse=True)
        
        best_score, best_candidate, best_data = candidate_scores[0]
        
        test_route = RouteBuilder(route.cell_data[route.cells[0]], 0, route.fulfillment_center, distance_method)
        for cell_id in route.cells:
            if cell_id != route.cells[0]:
                test_route.add_cell(route.cell_data[cell_id])
        test_route.add_cell(best_data)
        
        if validate_enhanced_constraints(test_route, min_customers, min_distance, max_distance,
                                       max_clusters_per_route, max_customers_per_route, 
                                       compactness_weight):
            route.add_cell(best_data)
            used_cells.add(best_candidate)
        else:
            break
    
    return route

def generate_routes(h3_dataset, fulfillment_center, adjacency_graph, 
                   min_customers=40, min_distance=3, max_distance=7, 
                   distance_method='vertex', score_method='vertex',
                   max_clusters_per_route=8, max_customers_per_route=200,
                   compactness_weight=0.3):
    routes = []
    used_cells = set()
    route_id = 0
    
    score_column = f'score_{score_method}'
    h3_sorted = h3_dataset.sort_values(score_column, ascending=False)
    
    for _, seed_cell in h3_sorted.iterrows():
        if seed_cell['h3_cell_id'] in used_cells:
            continue
        
        route_id += 1
        current_route = RouteBuilder(seed_cell, route_id, fulfillment_center, distance_method)
        used_cells.add(seed_cell['h3_cell_id'])
        
        expanded_route = expand_route_greedily(
            current_route, h3_dataset, adjacency_graph, used_cells,
            min_customers, min_distance, max_distance, distance_method, score_method,
            max_clusters_per_route, max_customers_per_route, compactness_weight
        )
        
        if validate_enhanced_constraints(expanded_route, min_customers, min_distance, max_distance,
                                       max_clusters_per_route, max_customers_per_route, 
                                       compactness_weight):
            routes.append(expanded_route)
            for cell_id in expanded_route.get_cell_ids():
                used_cells.add(cell_id)
    
    return routes

def calculate_comprehensive_route_metrics(route, fulfillment_center):
    fc_lat, fc_lon = fulfillment_center['latitude'], fulfillment_center['longitude']
    
    cumulative_distance = route.get_cumulative_distance()
    
    farthest_centroid = max([
        calculate_h3_centroid_distance(cell_id, fc_lat, fc_lon)
        for cell_id in route.get_cell_ids()
    ])
    
    farthest_vertex = max([
        calculate_h3_farthest_vertex_distance(cell_id, fc_lat, fc_lon)
        for cell_id in route.get_cell_ids()
    ])
    
    return {
        'cumulative_distance_km': round(cumulative_distance, 2),
        'farthest_centroid_distance_km': round(farthest_centroid, 2),
        'farthest_vertex_distance_km': round(farthest_vertex, 2)
    }

def create_output_dataframe(routes, df_output_assignment, fulfillment_center):
    output_rows = []
    
    for route in routes:
        route_metrics = calculate_comprehensive_route_metrics(route, fulfillment_center)
        
        route_customer_data = df_output_assignment[
            df_output_assignment['h3_cell_id'].isin(route.get_cell_ids())
        ]
        
        avg_confidence = route_customer_data['assignment_confidence'].mean()
        tier_summary = route_customer_data['assignment_tier'].value_counts().to_dict()
        
        output_rows.append({
            'route_id': route.route_id,
            'h3_cell_ids': route.get_cell_ids(),
            'cumulative_distance_km': route_metrics['cumulative_distance_km'],
            'farthest_centroid_distance_km': route_metrics['farthest_centroid_distance_km'],
            'farthest_vertex_distance_km': route_metrics['farthest_vertex_distance_km'],
            'customer_count': route.total_customers,
            'cluster_count': len(route.get_cell_ids()),
            'compactness_score': calculate_route_compactness(route.get_cell_ids(), fulfillment_center),
            'avg_assignment_confidence': round(avg_confidence, 3),
            'assignment_tier_summary': tier_summary
        })
    
    return pd.DataFrame(output_rows)

def main_route_planning_pipeline(sp_dim_df, customers_gdf, df_output_assignment, 
                                stock_point_id, w1=1.0, w2=1.0, 
                                min_customers=40, min_distance=3, max_distance=7, 
                                distance_method='vertex', score_method='vertex',
                                max_clusters_per_route=8, max_customers_per_route=200,
                                compactness_weight=0.3):
    print("🚀 Starting Enhanced Delivery Route Planning Pipeline")
    
    print("📊 Phase 1: Data Validation and Preparation")
    validate_input_data(sp_dim_df, customers_gdf, df_output_assignment)
    fulfillment_center = extract_fulfillment_center(sp_dim_df, stock_point_id)
    
    stock_assignments = df_output_assignment[
        df_output_assignment['stock_point_id'] == stock_point_id
    ]
    if stock_assignments.empty:
        raise ValueError(f"No assignments found for stock_point_id: {stock_point_id}")
    
    print("🔍 Phase 2: H3 Cell Analysis and Metrics Calculation")
    h3_dataset = build_h3_dataset(stock_assignments, fulfillment_center)
    h3_dataset = add_scores_to_dataset(h3_dataset, w1, w2, compactness_weight)
    
    print("🕸️ Phase 3: Adjacency Graph Construction")
    adjacency_graph = build_adjacency_graph(h3_dataset)
    
    print("🛣️ Phase 4: Enhanced Route Generation")
    routes = generate_routes(
        h3_dataset, fulfillment_center, adjacency_graph,
        min_customers, min_distance, max_distance,
        distance_method, score_method,
        max_clusters_per_route, max_customers_per_route, compactness_weight
    )
    
    print("📋 Phase 5: Output Generation")
    output_df = create_output_dataframe(routes, stock_assignments, fulfillment_center)
    
    print(f"✅ Generated {len(routes)} routes")
    print(f"📈 Total customers covered: {output_df['customer_count'].sum()}")
    print(f"📏 Average route distance: {output_df['farthest_vertex_distance_km'].mean():.2f} km")
    print(f"🔢 Average clusters per route: {output_df['cluster_count'].mean():.1f}")
    print(f"📊 Average compactness score: {output_df['compactness_score'].mean():.2f}")
    
    return output_df, routes, h3_dataset


def validate_route_quality(output_df, quality_thresholds=None):
    """
    Validate route quality against business standards and flag problematic routes
    """
    if quality_thresholds is None:
        quality_thresholds = {
            'max_compactness_score': 2.0,
            'min_customer_density': 5.0,  # customers per cluster
            'max_distance_variance': 1.5,  # km variance between routes
            'min_confidence_score': 0.8
        }
    
    quality_issues = []
    route_quality_scores = []
    
    for idx, route in output_df.iterrows():
        issues = []
        quality_score = 100  # Start with perfect score
        
        # Check compactness
        if route['compactness_score'] > quality_thresholds['max_compactness_score']:
            issues.append('High spatial spread')
            quality_score -= 20
        
        # Check customer density efficiency
        customer_density = route['customer_count'] / route['cluster_count']
        if customer_density < quality_thresholds['min_customer_density']:
            issues.append('Low customer density')
            quality_score -= 15
        
        # Check assignment confidence
        if route['avg_assignment_confidence'] < quality_thresholds['min_confidence_score']:
            issues.append('Low assignment confidence')
            quality_score -= 25
        
        # Check for single-cluster routes (potentially inefficient)
        if route['cluster_count'] == 1:
            issues.append('Single cluster route')
            quality_score -= 10
        
        # Check distance efficiency
        distance_per_customer = route['farthest_vertex_distance_km'] / route['customer_count']
        if distance_per_customer > 0.15:  # 150m per customer threshold
            issues.append('High distance per customer')
            quality_score -= 15
        
        quality_issues.append(issues if issues else ['No issues'])
        route_quality_scores.append(max(0, quality_score))
    
    # Add quality metrics to dataframe
    output_df['quality_issues'] = quality_issues
    output_df['quality_score'] = route_quality_scores
    
    # Generate summary
    avg_quality = np.mean(route_quality_scores)
    problematic_routes = len([score for score in route_quality_scores if score < 70])
    
    quality_summary = {
        'total_routes': len(output_df),
        'average_quality_score': round(avg_quality, 1),
        'problematic_routes_count': problematic_routes,
        'problematic_routes_percentage': round((problematic_routes / len(output_df)) * 100, 1),
        'routes_needing_attention': output_df[output_df['quality_score'] < 70]['route_id'].tolist()
    }
    
    return output_df, quality_summary

def generate_route_summary_statistics(output_df, fulfillment_center_info=None):
    """
    Generate comprehensive summary statistics for route analysis
    """
    if output_df.empty:
        return {"error": "No routes to analyze"}
    
    # Basic route statistics
    basic_stats = {
        'route_count': len(output_df),
        'total_customers_covered': output_df['customer_count'].sum(),
        'total_clusters_used': output_df['cluster_count'].sum(),
        'average_customers_per_route': round(output_df['customer_count'].mean(), 1),
        'average_clusters_per_route': round(output_df['cluster_count'].mean(), 1)
    }
    
    # Distance analysis
    distance_stats = {
        'average_route_distance': round(output_df['farthest_vertex_distance_km'].mean(), 2),
        'min_route_distance': round(output_df['farthest_vertex_distance_km'].min(), 2),
        'max_route_distance': round(output_df['farthest_vertex_distance_km'].max(), 2),
        'distance_std_deviation': round(output_df['farthest_vertex_distance_km'].std(), 2),
        'average_cumulative_distance': round(output_df['cumulative_distance_km'].mean(), 2)
    }
    
    # Efficiency metrics
    efficiency_stats = {
        'average_customers_per_km': round(output_df['customer_count'].sum() / output_df['farthest_vertex_distance_km'].sum(), 1),
        'average_compactness_score': round(output_df['compactness_score'].mean(), 2),
        'routes_within_distance_constraints': len(output_df[(output_df['farthest_vertex_distance_km'] >= 3) & 
                                                           (output_df['farthest_vertex_distance_km'] <= 7)]),
        'constraint_compliance_percentage': round((len(output_df[(output_df['farthest_vertex_distance_km'] >= 3) & 
                                                                (output_df['farthest_vertex_distance_km'] <= 7)]) / len(output_df)) * 100, 1)
    }
    
    # Quality distribution
    if 'quality_score' in output_df.columns:
        quality_distribution = {
            'excellent_routes': len(output_df[output_df['quality_score'] >= 90]),
            'good_routes': len(output_df[(output_df['quality_score'] >= 70) & (output_df['quality_score'] < 90)]),
            'fair_routes': len(output_df[(output_df['quality_score'] >= 50) & (output_df['quality_score'] < 70)]),
            'poor_routes': len(output_df[output_df['quality_score'] < 50])
        }
    else:
        quality_distribution = {"note": "Quality scores not available - run validate_route_quality first"}
    
    # Assignment confidence analysis
    confidence_stats = {
        'average_assignment_confidence': round(output_df['avg_assignment_confidence'].mean(), 3),
        'min_assignment_confidence': round(output_df['avg_assignment_confidence'].min(), 3),
        'max_assignment_confidence': round(output_df['avg_assignment_confidence'].max(), 3),
        'high_confidence_routes': len(output_df[output_df['avg_assignment_confidence'] >= 0.9])
    }
    
    # Tier analysis
    all_tiers = []
    for tier_summary in output_df['assignment_tier_summary']:
        if isinstance(tier_summary, dict):
            all_tiers.extend(tier_summary.keys())
    
    unique_tiers = list(set(all_tiers))
    tier_stats = {
        'unique_tiers_found': unique_tiers,
        'routes_with_mixed_tiers': len([ts for ts in output_df['assignment_tier_summary'] 
                                       if isinstance(ts, dict) and len(ts) > 1])
    }
    
    # Optimization opportunities
    optimization_opportunities = []
    
    if distance_stats['distance_std_deviation'] > 1.0:
        optimization_opportunities.append("High distance variance - consider rebalancing routes")
    
    if efficiency_stats['average_customers_per_km'] < 10:
        optimization_opportunities.append("Low customer density - routes may be too spread out")
    
    if basic_stats['average_clusters_per_route'] < 3:
        optimization_opportunities.append("Low cluster utilization - consider merging small routes")
    
    if confidence_stats['average_assignment_confidence'] < 0.85:
        optimization_opportunities.append("Low assignment confidence - review customer clustering")
    
    return {
        'basic_statistics': basic_stats,
        'distance_analysis': distance_stats,
        'efficiency_metrics': efficiency_stats,
        'quality_distribution': quality_distribution,
        'confidence_analysis': confidence_stats,
        'tier_analysis': tier_stats,
        'optimization_opportunities': optimization_opportunities,
        'generation_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def optimize_parameters(sp_dim_df, customers_gdf, df_output_assignment, stock_point_id, 
                       parameter_ranges=None, optimization_metric='efficiency'):
    """
    Find optimal parameters through systematic testing of different combinations
    """
    if parameter_ranges is None:
        parameter_ranges = {
            'w1': [0.8, 1.0, 1.2],
            'w2': [0.8, 1.0, 1.2],
            'max_clusters_per_route': [6, 8, 10],
            'max_customers_per_route': [150, 200, 250],
            'compactness_weight': [0.2, 0.3, 0.4]
        }
    
    optimization_results = []
    best_score = -float('inf')
    best_params = None
    
    print(f"🔍 Testing {np.prod([len(values) for values in parameter_ranges.values()])} parameter combinations...")
    
    total_combinations = 0
    successful_combinations = 0
    
    # Generate all combinations
    import itertools
    param_names = list(parameter_ranges.keys())
    param_combinations = list(itertools.product(*[parameter_ranges[param] for param in param_names]))
    
    for i, param_values in enumerate(param_combinations):
        total_combinations += 1
        params = dict(zip(param_names, param_values))
        
        try:
            # Run pipeline with current parameters
            from main_route_planning_pipeline import main_route_planning_pipeline
            
            output_df, routes, _ = main_route_planning_pipeline(
                sp_dim_df, customers_gdf, df_output_assignment, stock_point_id,
                w1=params['w1'], w2=params['w2'],
                max_clusters_per_route=params['max_clusters_per_route'],
                max_customers_per_route=params['max_customers_per_route'],
                compactness_weight=params['compactness_weight']
            )
            
            if len(output_df) == 0:
                continue
                
            successful_combinations += 1
            
            # Calculate optimization metrics
            metrics = calculate_optimization_metrics(output_df, optimization_metric)
            
            result = {
                'parameters': params,
                'metrics': metrics,
                'route_count': len(output_df),
                'total_customers': output_df['customer_count'].sum(),
                'avg_distance': output_df['farthest_vertex_distance_km'].mean(),
                'avg_compactness': output_df['compactness_score'].mean()
            }
            
            optimization_results.append(result)
            
            # Check if this is the best combination
            if metrics['composite_score'] > best_score:
                best_score = metrics['composite_score']
                best_params = params.copy()
                
        except Exception as e:
            print(f"Failed combination {i+1}: {params} - Error: {str(e)}")
            continue
    
    # Sort results by composite score
    optimization_results.sort(key=lambda x: x['metrics']['composite_score'], reverse=True)
    
    optimization_summary = {
        'total_combinations_tested': total_combinations,
        'successful_combinations': successful_combinations,
        'success_rate_percentage': round((successful_combinations / total_combinations) * 100, 1),
        'best_parameters': best_params,
        'best_score': round(best_score, 2),
        'top_5_results': optimization_results[:5]
    }
    
    return optimization_summary, optimization_results

def calculate_optimization_metrics(output_df, optimization_metric='efficiency'):
    """
    Calculate composite optimization score based on multiple factors
    """
    if len(output_df) == 0:
        return {'composite_score': 0, 'components': {}}
    
    # Normalize metrics to 0-100 scale for comparison
    
    # Efficiency: customers per km
    total_customers = output_df['customer_count'].sum()
    total_distance = output_df['farthest_vertex_distance_km'].sum()
    efficiency_score = min(100, (total_customers / total_distance) * 5)  # Scale factor
    
    # Coverage: percentage of customers covered (assuming target coverage)
    coverage_score = min(100, len(output_df) * 10)  # More routes = better coverage
    
    # Compactness: lower is better, invert and scale
    avg_compactness = output_df['compactness_score'].mean()
    compactness_score = max(0, 100 - (avg_compactness * 25))
    
    # Distance compliance: routes within 3-7km range
    compliant_routes = len(output_df[(output_df['farthest_vertex_distance_km'] >= 3) & 
                                   (output_df['farthest_vertex_distance_km'] <= 7)])
    compliance_score = (compliant_routes / len(output_df)) * 100
    
    # Route balance: low standard deviation in route sizes
    customer_std = output_df['customer_count'].std()
    balance_score = max(0, 100 - customer_std)
    
    # Composite score based on optimization focus
    if optimization_metric == 'efficiency':
        composite_score = (efficiency_score * 0.4 + compactness_score * 0.3 + 
                          compliance_score * 0.2 + balance_score * 0.1)
    elif optimization_metric == 'coverage':
        composite_score = (coverage_score * 0.4 + efficiency_score * 0.3 + 
                          compliance_score * 0.2 + compactness_score * 0.1)
    elif optimization_metric == 'compactness':
        composite_score = (compactness_score * 0.5 + compliance_score * 0.3 + 
                          efficiency_score * 0.2)
    else:  # balanced
        composite_score = (efficiency_score * 0.25 + coverage_score * 0.25 + 
                          compactness_score * 0.25 + compliance_score * 0.25)
    
    return {
        'composite_score': round(composite_score, 2),
        'components': {
            'efficiency_score': round(efficiency_score, 2),
            'coverage_score': round(coverage_score, 2),
            'compactness_score': round(compactness_score, 2),
            'compliance_score': round(compliance_score, 2),
            'balance_score': round(balance_score, 2)
        }
    }