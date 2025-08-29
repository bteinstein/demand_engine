USE VCONNECTMASTERDWR;

-- Example SQL for SQL Server 2022 (adjust data types and constraints as necessary)

CREATE TABLE ds_gis_stock_points (
    id INT PRIMARY KEY IDENTITY(1,1),
    stock_point_id INT NOT NULL,
    stock_point_name NVARCHAR(255) NOT NULL,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    created_at DATETIME DEFAULT GETDATE()
);

--- LGA GEOMETRY - Synergy between Business and Nigeria Data
DROP TABLE IF EXISTS ds_gis_lgas;
CREATE TABLE ds_gis_lgas (
    id INT PRIMARY KEY IDENTITY(1,1),
    lga_id INT NOT NULL,
    lgacode INT NOT NULL,
    lga_name NVARCHAR(255) NOT NULL,
    lga_name_ng NVARCHAR(255) NOT NULL,
    state_id INT NOT NULL,
    state_name NVARCHAR(255) NOT NULL,
    -- GEOMETRY type for spatial data. Ensure SQL Server Spatial types are enabled.
    geometry GEOMETRY,
    area_km2 DECIMAL(18, 2),
    population_density DECIMAL(18, 2)
);

CREATE TABLE ds_gis_stock_point_lgas_map (
    stock_point_id INT NOT NULL,
    lga_id INT NOT NULL,
    territory_version INT NOT NULL DEFAULT 1, -- Added for versioning
    PRIMARY KEY (stock_point_id, lga_id),
    FOREIGN KEY (stock_point_id) REFERENCES ds_gis_stock_points(stock_point_id),
    FOREIGN KEY (lga_id) REFERENCES ds_gis_lgas(lga_id)
);

CREATE TABLE ds_gis_customers (
    id INT PRIMARY KEY IDENTITY(1,1),
    contact_id INT NOT NULL,
    customer_id INT NOT NULL,
    customer_name NVARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    lga_id INT,
    stock_point_id INT,
    created_at DATETIME DEFAULT GETDATE(),
    location_status NVARCHAR(50), -- e.g., 'verified', 'unverified', 'manual_review'
    FOREIGN KEY (lga_id) REFERENCES ds_gis_lgas(lga_id),
    FOREIGN KEY (stock_point_id) REFERENCES ds_gis_stock_points(stock_point_id)
);

-- These tables will be populated later during clustering phases
CREATE TABLE ds_gis_h3_clusters (
    id INT PRIMARY KEY IDENTITY(1,1),
    stock_point_id INT NOT NULL,
    h3_resolution INT NOT NULL,
    -- Storing H3 cell IDs as NVARCHAR(16) for individual hex IDs
    -- Or use a text/array type for h3_cells[] for multiple base hex IDs if a cluster
    -- represents a collection of H3 cells. For now, we'll assume a cluster
    -- is primarily defined by a collection of H3 cells, so we might store them
    -- in a separate linking table or as a JSON array if SQL Server allows easily.
    -- Let's consider `h3_cells_json` for now and refine if needed.
    h3_cells_json NVARCHAR(MAX), -- Store as JSON array of H3 hex IDs
    centroid_lat DECIMAL(10, 8),
    centroid_lng DECIMAL(11, 8),
    customer_count INT,
    parent_cluster_id INT, -- For splitting support
    territory_version INT NOT NULL, -- To link to the territory definition version
    FOREIGN KEY (stock_point_id) REFERENCES ds_gis_stock_points(stock_point_id)
);

CREATE TABLE ds_gis_customer_clusters (
    customer_id INT NOT NULL,
    cluster_id INT NOT NULL,
    h3_cell_id NVARCHAR(16) NOT NULL, -- The specific H3 cell ID the customer falls into
    assignment_confidence DECIMAL(3,2), -- e.g., 1.0, 0.8, 0.0
    PRIMARY KEY (customer_id), -- Assuming a customer belongs to only one cluster
    FOREIGN KEY (customer_id) REFERENCES ds_gis_customers(customer_id),
    FOREIGN KEY (cluster_id) REFERENCES ds_gis_h3_clusters(id)
);