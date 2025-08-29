import geopandas as gpd
from shapely.geometry import Point, Polygon
from sqlalchemy import create_engine, text
import pandas as pd # For stock points and potentially LGA data

# --- Database Connection (SQL Server) ---
# Replace with your actual connection string
# For SQL Server, typically 'mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server'
# Make sure you have the ODBC driver installed.
DB_CONNECTION_STRING = "mssql+pyodbc://your_user:your_password@your_server/your_database?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DB_CONNECTION_STRING)

# --- 1. Ingest LGA Data ---
# Assuming 'nigeria_lgas.geojson' is your GeoJSON file
lgas_geojson_path = 'path/to/your/nigeria_lgas.geojson'
try:
    lgas_gdf = gpd.read_file(lgas_geojson_path)
    print(f"Loaded {len(lgas_gdf)} LGAs from GeoJSON.")

    # Prepare data for insertion
    lgas_to_insert = []
    for index, row in lgas_gdf.iterrows():
        lga_name = row['LGA_NAME'] # Adjust column names based on your GeoJSON
        state_name = row['STATE_NAME'] # Adjust column names
        geometry_wkt = row.geometry.wkt # Convert shapely geometry to WKT
        
        # Placeholder for area_km2 and population_density if not in GeoJSON
        # You will need to calculate or source these accurately later.
        area_km2 = row.geometry.area / 10**6 if row.geometry.is_valid else 0 # Simple area approx (degrees to km2 is complex, use accurate projection if available)
        population_density = 0 # Placeholder

        lgas_to_insert.append({
            'name': lga_name,
            'state': state_name,
            'geometry': geometry_wkt, # SQL Server will need to convert WKT to GEOMETRY type
            'area_km2': area_km2,
            'population_density': population_density
        })
    
    # Batch insert into the lgas table
    with engine.connect() as connection:
        for lga_data in lgas_to_insert:
            # Use text() to execute raw SQL, ensuring GEOMETRY conversion if needed
            # For SQL Server, you might need ST_GeomFromText()
            insert_sql = text("""
                INSERT INTO lgas (name, state, geometry, area_km2, population_density)
                VALUES (:name, :state, geometry::STGeomFromText(:geometry, 4326), :area_km2, :population_density)
            """)
            connection.execute(insert_sql, lga_data)
        connection.commit()
    print("LGA data inserted successfully.")

except FileNotFoundError:
    print(f"Error: LGA GeoJSON file not found at {lgas_geojson_path}")
except Exception as e:
    print(f"Error during LGA data ingestion: {e}")


# --- 2. Ingest Stock Point Data ---
# Assuming 'stock_points.csv' contains your stock point data
stock_points_csv_path = 'path/to/your/stock_points.csv'
try:
    stock_points_df = pd.read_csv(stock_points_csv_path)
    print(f"Loaded {len(stock_points_df)} stock points from CSV.")

    # Insert into the stock_points table
    # Using pandas to_sql for simpler DataFrame insertion, ensure table name matches
    stock_points_df[['name', 'latitude', 'longitude']].to_sql(
        'stock_points',
        con=engine,
        if_exists='append', # Append to existing table
        index=False
    )
    print("Stock point data inserted successfully.")

except FileNotFoundError:
    print(f"Error: Stock points CSV file not found at {stock_points_csv_path}")
except Exception as e:
    print(f"Error during stock point data ingestion: {e}")