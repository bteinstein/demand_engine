def phase_3_improved_clustering(self):
    """Phase 3: Advanced Geographic Clustering with Multiple Algorithms - FIXED"""
    print("\nPhase 3: Advanced Geographic Clustering")
    print("-" * 50)
    
    # Add enhanced compactness features
    self.calculate_enhanced_compactness_features()
    
    # Prepare features for clustering with NaN handling
    features = self.h3_metrics[[
        'centroid_lat', 'centroid_lon', 'distance_from_fc', 
        'density_score', 'priority_score', 'avg_neighbor_distance'
    ]].copy()
    
    # FIX: Handle NaN values before clustering
    print(f"  Checking for NaN values in features...")
    nan_counts = features.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"  Found NaN values: {dict(nan_counts[nan_counts > 0])}")
        
        # Fill NaN values with appropriate defaults
        features['avg_neighbor_distance'] = features['avg_neighbor_distance'].fillna(
            features['avg_neighbor_distance'].median()
        )
        features['priority_score'] = features['priority_score'].fillna(
            features['priority_score'].median()
        )
        features['density_score'] = features['density_score'].fillna(
            features['density_score'].median()
        )
        
        # For any remaining NaN values, use forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Final check: if still NaN, fill with column means
        if features.isnull().sum().sum() > 0:
            features = features.fillna(features.mean())
            
        print(f"  ✓ NaN values handled")
    
    # Verify no NaN values remain
    assert not features.isnull().any().any(), "NaN values still present after cleaning"
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Verify scaled features don't contain NaN
    if np.isnan(features_scaled).any():
        print("  WARNING: NaN values in scaled features, using robust scaling")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
    
    # Estimate number of routes needed
    total_customers = self.h3_metrics['customer_count'].sum()
    avg_customers_per_route = (self.min_customers + self.max_customers) / 2
    estimated_routes = max(1, int(total_customers / avg_customers_per_route))
    
    print(f"✓ Total customers: {total_customers}")
    print(f"✓ Estimated routes needed: {estimated_routes}")
    
    # Try multiple clustering approaches
    clustering_results = {}
    
    # 1. K-means clustering
    print("  Testing K-means clustering...")
    try:
        kmeans = KMeans(n_clusters=estimated_routes, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        clustering_results['kmeans'] = kmeans_labels
    except Exception as e:
        print(f"  K-means failed: {str(e)}")
        # Fallback: use simple distance-based clustering
        clustering_results['kmeans'] = self._fallback_distance_clustering(estimated_routes)
    
    # 2. DBSCAN clustering  
    print("  Testing DBSCAN clustering...")
    try:
        coords = features[['centroid_lat', 'centroid_lon']].values
        # Verify coords don't contain NaN
        if np.isnan(coords).any():
            coords = np.nan_to_num(coords, nan=np.nanmean(coords))
            
        distances = pdist(coords)
        eps_adaptive = np.percentile(distances, 15)  # More conservative
        min_samples = max(2, int(self.min_customers / self.h3_metrics['customer_count'].mean()))
        
        dbscan = DBSCAN(eps=eps_adaptive, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(coords)
        
        # Handle noise points
        if -1 in dbscan_labels:
            dbscan_labels = self._assign_noise_points(dbscan_labels, coords)
        
        clustering_results['dbscan'] = dbscan_labels
    except Exception as e:
        print(f"  DBSCAN failed: {str(e)}")
        clustering_results['dbscan'] = self._fallback_distance_clustering(estimated_routes)
    
    # 3. Priority-based clustering (new approach)
    print("  Testing priority-based clustering...")
    try:
        priority_labels = self._priority_based_clustering()
        clustering_results['priority'] = priority_labels
    except Exception as e:
        print(f"  Priority-based clustering failed: {str(e)}")
        clustering_results['priority'] = self._fallback_distance_clustering(estimated_routes)
    
    # Evaluate all clustering methods
    best_method = None
    best_score = -1
    
    for method, labels in clustering_results.items():
        self.h3_metrics[f'{method}_cluster'] = labels
        score = self._evaluate_clustering(f'{method}_cluster')
        print(f"  {method.capitalize()} clustering score: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_method = method
    
    print(f"✓ Best clustering method: {best_method.upper()} (score: {best_score:.3f})")
    
    # Use best clustering method
    self.h3_metrics['route_cluster'] = self.h3_metrics[f'{best_method}_cluster']
    
    # Clean up temporary columns
    temp_cols = [f'{method}_cluster' for method in clustering_results.keys()]
    self.h3_metrics = self.h3_metrics.drop(temp_cols, axis=1)
    
    return self.h3_metrics


def _fallback_distance_clustering(self, n_clusters):
    """Fallback clustering method based on distance from fulfillment center"""
    print("  Using fallback distance-based clustering...")
    
    # Sort by distance and assign clusters in round-robin fashion
    sorted_indices = self.h3_metrics['distance_from_fc'].argsort()
    labels = np.zeros(len(self.h3_metrics))
    
    for i, idx in enumerate(sorted_indices):
        labels[idx] = i % n_clusters
    
    return labels.astype(int)


def calculate_enhanced_compactness_features(self):
    """Calculate enhanced features for geographic compactness - FIXED"""
    coords = self.h3_metrics[['centroid_lat', 'centroid_lon']].values
    
    # Verify coordinates don't contain NaN
    if np.isnan(coords).any():
        print("  WARNING: NaN values in coordinates, cleaning...")
        coords = np.nan_to_num(coords, nan=0)
    
    # Calculate distance matrix between H3 cells
    try:
        dist_matrix = pdist(coords, metric='euclidean')
        dist_matrix_square = squareform(dist_matrix)
    except Exception as e:
        print(f"  Distance calculation failed: {str(e)}, using simplified approach")
        # Fallback: calculate simple distances
        n = len(coords)
        dist_matrix_square = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist_matrix_square[i, j] = np.sqrt(
                        (coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2
                    )
    
    # Calculate multiple compactness metrics
    k_neighbors = min(5, len(coords) - 1)
    compactness_metrics = []
    
    for i in range(len(coords)):
        distances = dist_matrix_square[i]
        distances = distances[distances > 0]  # Exclude self
        
        if len(distances) >= k_neighbors and k_neighbors > 0:
            avg_nearest = np.mean(np.sort(distances)[:k_neighbors])
            std_nearest = np.std(np.sort(distances)[:k_neighbors]) if k_neighbors > 1 else 0
            
            # Calculate convex hull ratio if enough points
            hull_ratio = 0  # Default value
            if len(distances) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    neighbor_indices = np.argsort(distances)[:k_neighbors]
                    neighbor_coords = coords[[i] + neighbor_indices.tolist()]
                    
                    # Clean coordinates for hull calculation
                    neighbor_coords = np.nan_to_num(neighbor_coords, nan=0)
                    
                    if len(np.unique(neighbor_coords, axis=0)) >= 3:  # Need at least 3 unique points
                        hull = ConvexHull(neighbor_coords)
                        hull_ratio = len(neighbor_coords) / hull.volume if hull.volume > 0 else 0
                except Exception:
                    hull_ratio = 0
        else:
            avg_nearest = np.mean(distances) if len(distances) > 0 else 0
            std_nearest = np.std(distances) if len(distances) > 0 else 0
            hull_ratio = 0
        
        # Handle potential NaN values
        avg_nearest = 0 if np.isnan(avg_nearest) else avg_nearest
        std_nearest = 0 if np.isnan(std_nearest) else std_nearest
        hull_ratio = 0 if np.isnan(hull_ratio) else hull_ratio
        
        compactness_metrics.append({
            'avg_neighbor_distance': avg_nearest,
            'std_neighbor_distance': std_nearest,
            'hull_compactness': hull_ratio
        })
    
    compactness_df = pd.DataFrame(compactness_metrics)
    
    # Final NaN check and cleaning
    for col in compactness_df.columns:
        compactness_df[col] = compactness_df[col].fillna(0)
        self.h3_metrics[col] = compactness_df[col]
    
    return self.h3_metrics


