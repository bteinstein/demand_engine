# ------------------------------------------------------------------------------------------------------
    # Customer Assignment
    # ------------------------------------------------------------------------------------------------------
    """
    MAIN TABLE NAME: customer_cluster_assignment
    1. Core Upsert Method
        ✅ Prevents duplicates by properly handling existing records
        ✅ Version control with change tracking
        ✅ Smart change detection - only updates when data actually changes
        ✅ Batch processing for performance
        ✅ Comprehensive logging and progress tracking

    2. Version Control System
        Status tracking: ACTIVE, SUPERSEDED, INACTIVE
        Temporal validity: valid_from/valid_to timestamps
        Change audit: previous values, change reason, who made changes
        Version numbering: Incremental versioning per customer

    3. Query & Analysis Methods
        get_customer_history() - Full timeline for a customer
        get_active_assignments() - Current state only
        get_change_summary() - Daily change statistics
        get_customer_movements() - Location/cluster changes
        check_duplicates() - Verify data integrity
        cleanup_old_records() - Archive management
        
    Usage Examples:
    from src.h3_spatial_system.h3_system.FastH3DuckDBManager import FastH3DuckDBManager
    H3_DUCKDB_PATH = STORAGE_CONFIG['h3_duckdb_path']

    # Your main usage - now with version control
    with FastH3DuckDBManager(resolution=8, db_path=H3_DUCKDB_PATH) as db: 
        # Upsert with change tracking
        db.upsert_customer_cluster_assignment(
            df_output_sp_customer_assignment,
            change_reason="CUSTOMER_LOCATION_UPDATE",
            changed_by="BATCH_PROCESSOR"
        )
        
        # Verify no duplicates
        db.check_duplicates()
        
        # Analyze recent changes
        changes = db.get_change_summary(days_back=7)
        movements = db.get_customer_movements(days_back=30)
        
        # Get specific customer history
        history = db.get_customer_history(customer_id=12345)
    """
        
    def _create_customer_cluster_assignment_table(self):
        """
        Create the customer_cluster_assignment table with full version control,
        change tracking, and performance indexes.
        
        Key Changes:
        - H3_resolution is now part of the business key
        - Unique constraint updated to include h3_resolution
        - Indexes updated to support resolution-aware queries
        
        Features:
        - Primary key auto-increment via sequence
        - Business data columns with h3_resolution as key component
        - Version control (status, version_number, temporal validity)
        - Change tracking (previous values, reason, user)
        - Resolution-aware indexes for active assignments
        """
        print("🏗️ Creating customer_cluster_assignment table with H3 resolution support...")

        # Create sequence for auto-increment ID
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS customer_stockpoint_cluster_assignment START 1;
        """)

        # Create table with unique constraint
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_stockpoint_cluster_assignment (
                -- Primary key
                id BIGINT PRIMARY KEY DEFAULT nextval('customer_cluster_assignment_id_seq'),
                
                -- Business data (h3_resolution is now a key business field)
                stock_point_id BIGINT NOT NULL,
                customer_id BIGINT NOT NULL,
                h3_resolution INT NOT NULL,
                cluster_id VARCHAR,
                h3_cell_id VARCHAR,
                assignment_confidence DOUBLE,
                assignment_tier VARCHAR,
                customer_type VARCHAR,
                
                -- Version Control & Change Tracking
                status VARCHAR DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INACTIVE', 'SUPERSEDED')),
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                valid_to TIMESTAMP DEFAULT NULL,
                version_number INTEGER DEFAULT 1,
                
                -- Change tracking
                previous_cluster_id VARCHAR,
                previous_h3_cell_id VARCHAR,
                previous_h3_resolution INT,
                previous_customer_type VARCHAR,
                change_reason VARCHAR,
                changed_by VARCHAR DEFAULT 'SYSTEM',
                
                -- Unique constraint for active assignments
                # UNIQUE(customer_id, stock_point_id, h3_resolution, status) # REMOVED status from unique constraint
                UNIQUE(customer_id, stock_point_id, h3_resolution)
            );
        """)

        # Create performance indexes
        self._create_customer_cluster_assignment_indexes()

        print("✅ Created customer_stockpoint_cluster_assignment table with H3 resolution support and indexes")
        
    def _add_customer_cluster_assignment_columns(self):
        """
        Ensure the customer_cluster_assignment table exists with version control columns
        and proper H3 resolution handling.
        
        Key Changes:
        - H3_resolution is now NOT NULL and part of business key
        - Added previous_h3_resolution for change tracking
        - Updated unique constraints to include h3_resolution
        
        Features:
        - Version control with status tracking (ACTIVE, SUPERSEDED, INACTIVE)
        - Temporal validity with valid_from/valid_to timestamps
        - Change tracking with previous values including resolution changes
        - Audit trail with change reasons
        - Resolution-aware constraints
        """
        # Check if table exists
        table_exists = self.conn.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'customer_stockpoint_cluster_assignment'
        """).fetchone()[0] > 0
        
        if not table_exists:
            print("🏗️ Table customer_cluster_assignment does not exist. Creating with H3 resolution support...")
            
            # Create sequence for auto-increment ID
            self._create_customer_cluster_assignment_table()            
        else:
            print("🔧 Table exists. Checking and adding H3 resolution support...")
            
            # Add version control columns if missing
            version_control_columns = [
                ("status", "VARCHAR DEFAULT 'ACTIVE'"),
                ("created_date", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("modified_date", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("valid_from", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
                ("valid_to", "TIMESTAMP DEFAULT NULL"),
                ("version_number", "INTEGER DEFAULT 1"),
                ("previous_cluster_id", "VARCHAR"),
                ("previous_h3_cell_id", "VARCHAR"),
                ("previous_h3_resolution", "INT"),  # NEW: Track resolution changes
                ("previous_customer_type", "VARCHAR"),  # NEW: Track customer changes
                ("change_reason", "VARCHAR"),
                ("changed_by", "VARCHAR DEFAULT 'SYSTEM'"),
            ]
            
            for col, col_type in version_control_columns:
                try:
                    self.conn.execute(
                        f"ALTER TABLE customer_cluster_assignment ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    )
                except Exception as e:
                    # Column might already exist
                    pass
            
            # Ensure base columns exist with proper h3_resolution handling
            base_columns = [
                ("stock_point_id", "BIGINT"),
                ("customer_id", "BIGINT"),
                ("h3_resolution", f"INT DEFAULT {self.resolution}"),  # Ensure default
                ("cluster_id", "VARCHAR"),
                ("h3_cell_id", "VARCHAR"),
                ("assignment_confidence", "DOUBLE"),
                ("assignment_tier", "VARCHAR"),
                ("customer_type", "VARCHAR") 
            ]

            for col, col_type in base_columns:
                try:
                    self.conn.execute(
                        f"ALTER TABLE customer_stockpoint_cluster_assignment ADD COLUMN IF NOT EXISTS {col} {col_type}"
                    )
                except Exception as e:
                    pass
            
            # CRITICAL: Make h3_resolution NOT NULL if it wasn't already
            try:
                # Update any NULL h3_resolution values first
                self.conn.execute(f"""
                    UPDATE customer_stockpoint_cluster_assignment 
                    SET h3_resolution = {self.resolution} 
                    WHERE h3_resolution IS NULL AND h3_cell_id IS NOT NULL
                """)
                
                # Now make it NOT NULL (DuckDB may not support this directly)
                print(f"ℹ️ Ensuring h3_resolution has default value {self.resolution} for existing records")
                
            except Exception as e:
                print(f"⚠️ Could not enforce NOT NULL on h3_resolution: {e}")
            
            # Update existing records without version control data
            self._migrate_existing_records()
            
            # Drop old unique constraint if it exists
            try:
                self.conn.execute("DROP INDEX IF EXISTS uk_customer_active_assignment;")
            except Exception as e:
                pass
            
            # Create indexes if they don't exist
            self._create_customer_cluster_assignment_indexes()
            
            print("✅ Schema updated with H3 resolution support and version control columns")

    def _create_customer_cluster_assignment_indexes(self):
        """Create performance indexes for the table with H3 resolution support"""
        indexes = [
            # Standard indexes (no WHERE clauses for DuckDB compatibility)
            ("idx_customer_assignment_active_res", "customer_id, stock_point_id, h3_resolution, status"),
            ("idx_customer_assignment_dates", "created_date, valid_from, valid_to"),
            ("idx_customer_assignment_customer", "customer_id"),
            ("idx_customer_assignment_cluster", "cluster_id"),
            ("idx_customer_assignment_h3", "h3_cell_id"),
            ("idx_customer_assignment_resolution", "h3_resolution"),
            ("idx_customer_assignment_stock_res", "stock_point_id, h3_resolution"),
            ("idx_customer_assignment_version", "version_number"),
            ("idx_customer_assignment_status", "status"),  # NEW: Status-specific queries
        ]
        
        for idx_name, columns in indexes:
            try:
                self.conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} 
                    ON customer_stockpoint_cluster_assignment({columns});
                """)
            except Exception as e:
                print(f"⚠️ Could not create index {idx_name}: {e}")
        
        print("✅ Created performance indexes")
        
    def _migrate_existing_records(self):
        """Update existing records with default version control values and h3_resolution"""
        try:
            current_time = datetime.now()
            
            # Update records that don't have version control data or h3_resolution
            self.conn.execute(f"""
                UPDATE customer_stockpoint_cluster_assignment 
                SET 
                    status = COALESCE(status, 'ACTIVE'),
                    created_date = COALESCE(created_date, '{current_time}'),
                    modified_date = COALESCE(modified_date, '{current_time}'),
                    valid_from = COALESCE(valid_from, '{current_time}'),
                    version_number = COALESCE(version_number, 1),
                    changed_by = COALESCE(changed_by, 'MIGRATION'),
                    h3_resolution = COALESCE(h3_resolution, {self.resolution}),
                    customer_type = COALESCE(customer_type, 'UNKNOWN'),
                WHERE status IS NULL OR version_number IS NULL OR h3_resolution IS NULL
            """)
            
            print("✅ Migrated existing records with version control defaults and h3_resolution")
            
        except Exception as e:
            print(f"⚠️ Migration of existing records failed: {e}")

    def upsert_customer_cluster_assignment_reviewing(self, df: pd.DataFrame, batch_size: int = 10000, 
                                         change_reason: str = "BATCH_UPDATE", changed_by: str = "SYSTEM"):
        """
        Version-controlled upsert that maintains historical records and prevents duplicates.
        Now properly handles H3 resolution as part of the business key.
        
        CRITICAL CHANGES:
        - H3_resolution is now part of the unique business key
        - Customers can have MULTIPLE ACTIVE records per stock_point_id (one per resolution)
        - Comparison logic updated to include h3_resolution
        - Change tracking includes resolution changes
        
        Logic:
        1. For existing customers at same resolution: Mark old records as 'SUPERSEDED', insert new as 'ACTIVE'
        2. For existing customers at different resolution: Insert as new 'ACTIVE' (multiple active allowed)
        3. For new customers: Insert as 'ACTIVE'
        4. Track what changed (cluster_id, h3_cell_id, resolution movements)
        5. Prevent duplicate active records per (customer_id, stock_point_id, h3_resolution)
        
        Args:
            df: DataFrame with customer assignment data
            batch_size: Number of records to process per batch
            change_reason: Reason for the change (e.g., 'RESOLUTION_CHANGE', 'REBALANCING', 'BATCH_UPDATE')
            changed_by: Who/what made the change (e.g., 'USER123', 'SYSTEM', 'API')
            
        Expected DataFrame columns:
            ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 
             'assignment_confidence', 'assignment_tier', 'h3_resolution']
        """
        if df.empty:
            print("⚠️ No data provided for customer cluster assignment")
            return
        
        print(f"📦 Upserting {len(df):,} records into customer_stockpoint_cluster_assignment with H3 resolution support...")
         
        # Check for duplicates FIRST, before adding metadata
        duplicates = df.duplicated(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep=False)
        if duplicates.any():
            dup_count = duplicates.sum()
            print(f"⚠️ Found {dup_count} duplicate records in input data - keeping last occurrence")
            df = df.drop_duplicates(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep='last')

        print(f"📦 Upserting {len(df):,} records...")
          
        # Ensure schema exists with version control and h3_resolution support
        self._add_customer_cluster_assignment_columns()
        
        # Validate required columns  
        required_cols = ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 'h3_resolution','customer_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate h3_resolution values are not null
        null_resolution_count = df['h3_resolution'].isnull().sum()
        if null_resolution_count > 0:
            print(f"⚠️ Found {null_resolution_count} records with null h3_resolution, using default {self.resolution}")
            df['h3_resolution'] = df['h3_resolution'].fillna(self.resolution)
        
        # Add metadata columns to input DataFrame
        current_timestamp = datetime.now()
        df_with_metadata = df.copy()
        df_with_metadata['status'] = 'ACTIVE'
        df_with_metadata['created_date'] = current_timestamp
        df_with_metadata['modified_date'] = current_timestamp
        df_with_metadata['valid_from'] = current_timestamp
        df_with_metadata['valid_to'] = None
        df_with_metadata['change_reason'] = change_reason
        df_with_metadata['changed_by'] = changed_by
        
        # Initialize counters
        total_processed = 0
        total_superseded = 0
        total_new = 0
        total_unchanged = 0
        inactive_count = 0
        
        successfully_processed_keys = []
        
        # Process in batches
        for i in range(0, len(df_with_metadata), batch_size):
            batch_df = df_with_metadata.iloc[i:i+batch_size].copy()
            batch_num = i // batch_size + 1
            
            try:
                self.conn.register('batch_df', batch_df)
                
                # CRITICAL CHANGE: Get existing active records for comparison INCLUDING h3_resolution
                existing_records = self.conn.execute("""
                    SELECT 
                        e.id, e.customer_id, e.stock_point_id, e.h3_resolution, e.customer_type,
                        e.cluster_id, e.h3_cell_id, e.assignment_confidence, 
                        e.assignment_tier, e.version_number, e.created_date
                    FROM customer_stockpoint_cluster_assignment e
                    INNER JOIN batch_df b ON (
                        e.customer_id = b.customer_id 
                        AND e.stock_point_id = b.stock_point_id
                        AND e.h3_resolution = b.h3_resolution
                    )
                    WHERE e.status = 'ACTIVE'
                """).fetchall()
                
                if existing_records: 
                    existing_df = pd.DataFrame(existing_records, columns=[
                        'existing_id', 'customer_id', 'stock_point_id', 'h3_resolution', 'existing_customer_type',  # Changed here
                        'existing_cluster_id', 'existing_h3_cell_id', 'existing_confidence', 
                        'existing_tier', 'existing_version', 'existing_created_date'
                    ])
                    
                    
                    # Merge with new data to identify changes
                    batch_with_existing = batch_df.merge(
                        existing_df, 
                        on=['customer_id', 'stock_point_id', 'h3_resolution'],   # REMOVED 'customer_type'
                        how='left'
                    )
                    
                    changed_mask = (
                        (batch_with_existing['cluster_id'].fillna('') != batch_with_existing['existing_cluster_id'].fillna('')) |
                        (batch_with_existing['h3_cell_id'].fillna('') != batch_with_existing['existing_h3_cell_id'].fillna('')) |
                        (batch_with_existing['assignment_confidence'].fillna(0) != batch_with_existing['existing_confidence'].fillna(0)) |
                        (batch_with_existing['assignment_tier'].fillna('') != batch_with_existing['existing_tier'].fillna('')) | 
                        (batch_with_existing['customer_type'].fillna('') != batch_with_existing['existing_customer_type'].fillna(''))
                    )
                    
                    
                    changed_records = batch_with_existing[changed_mask].copy()
                    unchanged_records = batch_with_existing[~changed_mask]
                    
                    unchanged_count = len(unchanged_records)
                    total_unchanged += unchanged_count
                    
                    if len(changed_records) > 0:
                        # Add change tracking information
                        changed_records['previous_cluster_id'] = changed_records['existing_cluster_id']
                        changed_records['previous_h3_cell_id'] = changed_records['existing_h3_cell_id']
                        changed_records['previous_h3_resolution'] = changed_records['h3_resolution']  # Track resolution
                        changed_records['previous_customer_type'] = changed_records['existing_customer_type']  # Track CUSTOMER TYPE
                        changed_records['version_number'] = changed_records['existing_version'] + 1
                        
                        # Register changed records for processing
                        self.conn.unregister('batch_df')
                        self.conn.register('changed_batch', changed_records)
                        
                        # UPDATED: Mark existing changed records as SUPERSEDED (including h3_resolution in match)
                        self.conn.execute(f"""
                            UPDATE customer_stockpoint_cluster_assignment 
                            SET 
                                status = 'SUPERSEDED',
                                valid_to = '{current_timestamp}',
                                modified_date = '{current_timestamp}'
                            WHERE (customer_id, stock_point_id, h3_resolution) IN (
                                SELECT customer_id, stock_point_id, h3_resolution FROM changed_batch
                            ) AND status = 'ACTIVE'
                        """)
                        
                        superseded_count = len(changed_records)
                        total_superseded += superseded_count
                        
                        # Step 3: Insert new ACTIVE records for changed data
                        self.conn.execute("""
                            INSERT INTO customer_stockpoint_cluster_assignment (
                                stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                                assignment_confidence, assignment_tier,
                                status, created_date, modified_date, valid_from, valid_to,
                                version_number, previous_cluster_id, previous_h3_cell_id, 
                                previous_h3_resolution, previous_customer_type, change_reason, changed_by
                            )
                            SELECT
                                stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                                assignment_confidence, assignment_tier,
                                status, created_date, modified_date, valid_from, valid_to,
                                version_number, previous_cluster_id, previous_h3_cell_id,
                                previous_h3_resolution, previous_customer_type, change_reason, changed_by
                            FROM changed_batch
                        """)
                        
                        self.conn.unregister('changed_batch')
                        total_new += len(changed_records)
                        
                        print(f"✅ Batch {batch_num}: {len(changed_records)} changed, {unchanged_count} unchanged, {superseded_count} superseded")
                    
                    else:
                        print(f"ℹ️ Batch {batch_num}: All {unchanged_count} records unchanged, skipping")
                
                else:
                    # No existing records, these are all new customers (or new resolutions)
                    batch_df['version_number'] = 1
                    batch_df['previous_cluster_id'] = None
                    batch_df['previous_h3_cell_id'] = None
                    batch_df['previous_h3_resolution'] = None
                    batch_df['previous_customer_type'] = None
                    
                    self.conn.unregister('batch_df')
                    self.conn.register('new_batch', batch_df)
                    
                    # Insert all as new ACTIVE records
                    self.conn.execute("""
                        INSERT INTO customer_stockpoint_cluster_assignment (
                            stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                            assignment_confidence, assignment_tier,
                            status, created_date, modified_date, valid_from, valid_to,
                            version_number, previous_cluster_id, previous_h3_cell_id, 
                            previous_h3_resolution, previous_customer_type, change_reason, changed_by
                        )
                        SELECT
                            stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                            assignment_confidence, assignment_tier,
                            status, created_date, modified_date, valid_from, valid_to,
                            version_number, previous_cluster_id, previous_h3_cell_id,
                            previous_h3_resolution, previous_customer_type, change_reason, changed_by
                        FROM new_batch
                    """)
                    
                    self.conn.unregister('new_batch')
                    batch_count = len(batch_df)
                    total_new += batch_count
                    
                    print(f"✅ Batch {batch_num}: {batch_count} new customer-resolution assignments added")
                
                total_processed += len(batch_df)
                
                batch_keys = batch_df[['customer_id', 'stock_point_id', 'h3_resolution']].to_dict('records')
                successfully_processed_keys.extend(batch_keys)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {e}")
                # Clean up any registered dataframes
                for df_name in ['batch_df', 'changed_batch', 'new_batch']:
                    try:
                        self.conn.unregister(df_name)
                    except:
                        pass
                continue
        
        
        
        print("🔍 Checking for records to mark as INACTIVE (not in current dataset)...")
        if successfully_processed_keys:
            success_df = pd.DataFrame(successfully_processed_keys)
            self.conn.register('complete_df', success_df)
        
            # Find and update records that are ACTIVE in DB but missing from current df
            inactive_result = self.conn.execute(f"""
                UPDATE customer_stockpoint_cluster_assignment 
                SET status = 'INACTIVE', 
                    valid_to = '{current_timestamp}',
                    modified_date = '{current_timestamp}',
                    change_reason = '{change_reason}_REMOVAL',
                    changed_by = '{changed_by}'
                WHERE status = 'ACTIVE' 
                AND (customer_id, stock_point_id, h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM complete_df
                )
            """)
            
            inactive_count = inactive_result.rowcount if hasattr(inactive_result, 'rowcount') else 0
            self.conn.unregister('complete_df')
            
            if inactive_count > 0:
                print(f"🔄 Marked {inactive_count} records as INACTIVE (removed from source)")
            else:
                print("ℹ️ No records to mark as INACTIVE")
        else:
            print("⚠️ No batches processed successfully, skipping inactive check")
            inactive_count = 0    
            
        # Final summary
        print(f"""
                🎉 Customer cluster assignment upsert completed:
                📊 Total processed: {total_processed:,} records
                🆕 New/updated records: {total_new:,}
                🔄 Superseded records: {total_superseded:,}  
                ⚡ Unchanged records: {total_unchanged:,}
                ❌ Inactive records: {inactive_count:,}
                📝 Change reason: {change_reason}
                👤 Changed by: {changed_by}
                🔧 H3 resolution support: ✅ ENABLED
                """)

    def upsert_customer_cluster_assignment(self, df: pd.DataFrame, 
                                        change_reason: str = "BATCH_UPDATE", changed_by: str = "SYSTEM"):
        """
        Efficient version-controlled upsert using MERGE-like operations.
        Processes entire dataset at once for maximum efficiency.
        """
        if df.empty:
            print("⚠️ No data provided")
            return

        print(f"📦 Processing {len(df):,} records...")
        
        # Ensure schema and validate
        self._add_customer_cluster_assignment_columns()
        required_cols = ['customer_id', 'stock_point_id', 'cluster_id', 'h3_cell_id', 'h3_resolution', 'customer_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Clean and prepare data
        df = df.fillna({'h3_resolution': self.resolution})
        df = df.drop_duplicates(subset=['customer_id', 'stock_point_id', 'h3_resolution'], keep='last')
        
        current_timestamp = datetime.now()
        
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.register('input_data', df)
            
            # Step 1: Archive existing ACTIVE records that will be updated
            archive_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date, 
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id, 
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    e.stock_point_id, e.customer_id, e.h3_resolution, e.customer_type, 
                    e.cluster_id, e.h3_cell_id, e.assignment_confidence, e.assignment_tier,
                    'SUPERSEDED', e.created_date, '{current_timestamp}', e.valid_from, '{current_timestamp}',
                    e.version_number, e.previous_cluster_id, e.previous_h3_cell_id,
                    e.previous_h3_resolution, e.previous_customer_type, 
                    COALESCE(e.change_reason, '{change_reason}'), COALESCE(e.changed_by, '{changed_by}')
                FROM customer_stockpoint_cluster_assignment e
                INNER JOIN input_data i ON (
                    e.customer_id = i.customer_id AND 
                    e.stock_point_id = i.stock_point_id AND 
                    e.h3_resolution = i.h3_resolution
                )
                WHERE e.status = 'ACTIVE'
                AND (
                    e.cluster_id != i.cluster_id OR
                    e.h3_cell_id != i.h3_cell_id OR
                    COALESCE(e.assignment_confidence, 0) != COALESCE(i.assignment_confidence, 0) OR
                    COALESCE(e.assignment_tier, '') != COALESCE(i.assignment_tier, '') OR
                    COALESCE(e.customer_type, '') != COALESCE(i.customer_type, '')
                )
            """).rowcount or 0
            
            # Step 2: Delete ACTIVE records being replaced
            delete_count = self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE' 
                AND (customer_id, stock_point_id, h3_resolution) IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """).rowcount or 0
            
            # Step 3: Insert all new ACTIVE records with version tracking
            insert_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date,
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id,
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    i.stock_point_id, i.customer_id, i.h3_resolution, i.customer_type, 
                    i.cluster_id, i.h3_cell_id, i.assignment_confidence, i.assignment_tier,
                    'ACTIVE', '{current_timestamp}', '{current_timestamp}', '{current_timestamp}', NULL,
                    COALESCE(e.version_number, 0) + 1,
                    e.cluster_id, e.h3_cell_id, e.h3_resolution, e.customer_type,
                    '{change_reason}', '{changed_by}'
                FROM input_data i
                LEFT JOIN (
                    SELECT customer_id, stock_point_id, h3_resolution, cluster_id, h3_cell_id, 
                        customer_type, version_number,
                        ROW_NUMBER() OVER (PARTITION BY customer_id, stock_point_id, h3_resolution ORDER BY id DESC) as rn
                    FROM customer_stockpoint_cluster_assignment 
                    WHERE status IN ('ACTIVE', 'SUPERSEDED')
                ) e ON (
                    i.customer_id = e.customer_id AND 
                    i.stock_point_id = e.stock_point_id AND 
                    i.h3_resolution = e.h3_resolution AND
                    e.rn = 1
                )
            """).rowcount or 0
            
            # Step 4: Archive records not in current dataset (mark as INACTIVE)
            inactive_count = self.conn.execute(f"""
                INSERT INTO customer_stockpoint_cluster_assignment (
                    stock_point_id, customer_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier, status, created_date, modified_date,
                    valid_from, valid_to, version_number, previous_cluster_id, previous_h3_cell_id,
                    previous_h3_resolution, previous_customer_type, change_reason, changed_by
                )
                SELECT DISTINCT
                    e.stock_point_id, e.customer_id, e.h3_resolution, e.customer_type,
                    e.cluster_id, e.h3_cell_id, e.assignment_confidence, e.assignment_tier,
                    'INACTIVE', e.created_date, '{current_timestamp}', e.valid_from, '{current_timestamp}',
                    e.version_number, e.previous_cluster_id, e.previous_h3_cell_id,
                    e.previous_h3_resolution, e.previous_customer_type,
                    '{change_reason}_REMOVAL', '{changed_by}'
                FROM customer_stockpoint_cluster_assignment e
                WHERE e.status = 'ACTIVE'
                AND (e.customer_id, e.stock_point_id, e.h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """).rowcount or 0
            
            # Step 5: Remove old ACTIVE records that are now INACTIVE
            self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                AND (customer_id, stock_point_id, h3_resolution) NOT IN (
                    SELECT customer_id, stock_point_id, h3_resolution FROM input_data
                )
            """)
            
            self.conn.execute("COMMIT")
            
            # Calculate unchanged records
            unchanged_count = max(0, len(df) - (insert_count - delete_count))
            
            print(f"""✅ Upsert completed:
    📊 Processed: {len(df):,} | New/Updated: {insert_count:,} | Unchanged: {unchanged_count:,}
    🔄 Superseded: {archive_count:,} | Inactive: {inactive_count:,}""")
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            print(f"❌ Error: {e}")
            raise
        finally:
            try:
                self.conn.unregister('input_data')
            except:
                pass
    
    def get_customer_history(self, customer_id: int, stock_point_id: int = None, h3_resolution: int = None):
        """
        Get the complete history of cluster assignments for a customer.
        Now supports filtering by H3 resolution.
        
        Args:
            customer_id: Customer ID to look up
            stock_point_id: Optional stock point filter
            h3_resolution: Optional H3 resolution filter
            
        Returns:
            pandas.DataFrame: Historical assignment records ordered by version (newest first)
        """
        where_clause = "WHERE customer_id = ?"
        params = [customer_id]
        
        if stock_point_id:
            where_clause += " AND stock_point_id = ?"
            params.append(stock_point_id)
            
        if h3_resolution:
            where_clause += " AND h3_resolution = ?"
            params.append(h3_resolution)
        
        try:
            result = self.conn.execute(f"""
                SELECT 
                    id, stock_point_id, h3_resolution,  customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier,
                    status, version_number,
                    created_date, modified_date, valid_from, valid_to,
                    previous_cluster_id, previous_h3_cell_id, previous_h3_resolution, previous_customer_type, 
                    change_reason, changed_by
                FROM customer_stockpoint_cluster_assignment
                {where_clause}
                ORDER BY h3_resolution, version_number DESC, created_date DESC
            """, params).fetchall()
            
            if result:
                columns = [
                    'id', 'stock_point_id', 'h3_resolution', 'customer_type', 'cluster_id', 'h3_cell_id', 
                    'assignment_confidence', 'assignment_tier',
                    'status', 'version_number', 'created_date', 'modified_date', 
                    'valid_from', 'valid_to', 'previous_cluster_id', 'previous_h3_cell_id',
                    'previous_h3_resolution', 'previous_customer_type', 'change_reason', 'changed_by'
                ]
                
                history_df = pd.DataFrame(result, columns=columns)
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else " (all resolutions)"
                print(f"📋 Found {len(history_df)} historical records for customer {customer_id}{resolution_info}")
                return history_df
            else:
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else ""
                print(f"❌ No records found for customer {customer_id}{resolution_info}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving customer history: {e}")
            return pd.DataFrame()

    def get_active_assignments(self, stock_point_id: int = None, h3_resolution: int = None):
        """
        Get all currently active customer assignments.
        Now supports filtering by H3 resolution.
        
        Args:
            stock_point_id: Optional stock point filter
            h3_resolution: Optional H3 resolution filter
            
        Returns:
            pandas.DataFrame: Currently active assignment records
        """
        where_clause = "WHERE status = 'ACTIVE'"
        params = []
        
        if stock_point_id:
            where_clause += " AND stock_point_id = ?"
            params.append(stock_point_id)
            
        if h3_resolution:
            where_clause += " AND h3_resolution = ?"
            params.append(h3_resolution)
        
        try:
            result = self.conn.execute(f"""
                SELECT 
                    customer_id, stock_point_id, h3_resolution, customer_type, cluster_id, h3_cell_id,
                    assignment_confidence, assignment_tier,
                    created_date, modified_date, version_number,
                    change_reason, changed_by
                FROM customer_stockpoint_cluster_assignment
                {where_clause}
                ORDER BY customer_id, stock_point_id, h3_resolution
            """, params).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution', 'customer_type', 'cluster_id', 'h3_cell_id',
                    'assignment_confidence', 'assignment_tier',
                    'created_date', 'modified_date', 'version_number',
                    'change_reason', 'changed_by'
                ]
                
                active_df = pd.DataFrame(result, columns=columns)
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else " (all resolutions)"
                print(f"📊 Found {len(active_df)} active assignments{resolution_info}")
                return active_df
            else:
                resolution_info = f" at resolution {h3_resolution}" if h3_resolution else ""
                print(f"❌ No active assignments found{resolution_info}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving active assignments: {e}")
            return pd.DataFrame()

    def get_resolution_summary(self):
        """
        NEW METHOD: Get summary of assignments by H3 resolution.
        Useful for understanding how customers are distributed across different resolutions.
        
        Returns:
            pandas.DataFrame: Summary by resolution including counts and status distribution
        """
        try:
            result = self.conn.execute("""
                SELECT 
                    h3_resolution,
                    status,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points,
                    COUNT(DISTINCT cluster_id) as unique_clusters,
                    AVG(assignment_confidence) as avg_confidence,
                    MIN(created_date) as earliest_assignment,
                    MAX(created_date) as latest_assignment
                FROM customer_stockpoint_cluster_assignment
                GROUP BY h3_resolution, status
                ORDER BY h3_resolution, status
            """).fetchall()
            
            if result:
                columns = [
                    'h3_resolution', 'status', 'record_count', 'unique_customers',
                    'unique_stock_points', 'unique_clusters', 'avg_confidence',
                    'earliest_assignment', 'latest_assignment'
                ]
                summary_df = pd.DataFrame(result, columns=columns)
                print(f"📈 H3 Resolution summary:")
                return summary_df
            else:
                print("❌ No resolution data found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving resolution summary: {e}")
            return pd.DataFrame()

    def get_customer_movements(self, days_back: int = 30, include_resolution_changes: bool = True):
        """
        Get customers who changed clusters/locations/resolutions in the specified period.
        Now tracks H3 resolution changes as well.
        
        Args:
            days_back: Number of days to look back
            include_resolution_changes: Whether to include resolution-only changes
            
        Returns:
            pandas.DataFrame: Customer movement records
        """
        print(f"\n{'-'*100}")
        try:
            # Build the change condition
            change_conditions = [
                "previous_cluster_id != cluster_id",
                "previous_h3_cell_id != h3_cell_id",
                "previous_customer_type !=  customer_type"
            ]
            
            if include_resolution_changes:
                change_conditions.append("previous_h3_resolution != h3_resolution")
            
            change_condition = " OR ".join(change_conditions)
            
            result = self.conn.execute(f"""
                SELECT 
                    customer_id, stock_point_id, h3_resolution,customer_type,
                    previous_cluster_id, cluster_id,
                    previous_h3_cell_id, h3_cell_id,
                    previous_h3_resolution, previous_customer_type,
                    assignment_confidence, assignment_tier,
                    created_date, change_reason, changed_by,
                    version_number
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                  AND previous_cluster_id IS NOT NULL
                  AND created_date >= CURRENT_DATE - INTERVAL {days_back} DAYS
                  AND ({change_condition})
                ORDER BY created_date DESC, customer_id
            """).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution','customer_type',
                    'previous_cluster_id', 'cluster_id', 'previous_h3_cell_id', 'h3_cell_id',
                    'previous_h3_resolution',  'previous_customer_type','assignment_confidence', 'assignment_tier', 
                    'created_date', 'change_reason', 'changed_by', 'version_number'
                ]
                
                movements_df = pd.DataFrame(result, columns=columns)
                resolution_info = " (including resolution changes)" if include_resolution_changes else " (excluding resolution-only changes)"
                print(f"🚶 Found {len(movements_df)} customer movements in last {days_back} days{resolution_info}")
                return movements_df
            else:
                print(f"❌ No customer movements found in last {days_back} days")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving customer movements: {e}")
            return pd.DataFrame()

    def check_duplicates(self):
        """
        Helper method to check for duplicate active records in customer_stockpoint_cluster_assignment.
        Updated to include h3_resolution in duplicate detection.
        Should return empty result if upsert logic is working correctly.
        
        Returns:
            pandas.DataFrame: Any duplicate active assignments found
        """
        try:
            result = self.conn.execute("""
                SELECT 
                    customer_id, stock_point_id, h3_resolution, customer_type,
                    COUNT(*) as active_count,
                    string_agg(CAST(id AS VARCHAR), ', ' ORDER BY id) as record_ids
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                GROUP BY customer_id, stock_point_id, h3_resolution, customer_type
                HAVING COUNT(*) > 1
                ORDER BY active_count DESC
                LIMIT 10
            """).fetchall()
            
            if result:
                print(f"⚠️ Found {len(result)} duplicate active assignments (same customer + stock_point + resolution):")
                columns = ['customer_id', 'stock_point_id', 'h3_resolution', 'active_count', 'record_ids']
                duplicates_df = pd.DataFrame(result, columns=columns)
                print(duplicates_df)
                return duplicates_df
            else:
                print("✅ No duplicate active assignments found!")
            
            # Additional stats with resolution breakdown
            resolution_stats = self.conn.execute("""
                SELECT 
                    h3_resolution,
                    COUNT(*) as active_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points
                FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'ACTIVE'
                GROUP BY h3_resolution
                ORDER BY h3_resolution
            """).fetchall()
            
            if resolution_stats:
                print(f"\n📊 Active records by H3 resolution:")
                for res, count, customers, stock_points in resolution_stats:
                    print(f"   Resolution {res}: {count:,} records ({customers:,} customers, {stock_points:,} stock points)")
            
            total_active = self.conn.execute("""
                SELECT COUNT(*) FROM customer_stockpoint_cluster_assignment WHERE status = 'ACTIVE'
            """).fetchone()[0]
            
            total_all = self.conn.execute("""
                SELECT COUNT(*) FROM customer_stockpoint_cluster_assignment
            """).fetchone()[0]
            
            # Check for customers with multiple active resolutions (this is now allowed)
            multi_resolution_customers = self.conn.execute("""
                SELECT 
                    customer_id, stock_point_id,
                    COUNT(DISTINCT h3_resolution) as resolution_count,
                    string_agg(CAST(h3_resolution AS VARCHAR), ', ' ORDER BY h3_resolution) as resolutions
                FROM customer_stockpoint_cluster_assignment
                WHERE status = 'ACTIVE'
                GROUP BY customer_id, stock_point_id
                HAVING COUNT(DISTINCT h3_resolution) > 1
                LIMIT 5
            """).fetchall()
            
            if multi_resolution_customers:
                print(f"\nℹ️ Customers with multiple active resolutions (this is normal):")
                for customer_id, stock_point_id, res_count, resolutions in multi_resolution_customers:
                    print(f"   Customer {customer_id} at stock point {stock_point_id}: {res_count} resolutions ({resolutions})")
            
            print(f"\n📊 Total active records: {total_active:,}")
            print(f"📊 Total all records: {total_all:,}")
            
            return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error checking duplicates: {e}")
            return pd.DataFrame()

    def get_resolution_conflicts(self):
        """
        NEW METHOD: Identify potential conflicts where customers have assignments
        at multiple resolutions that might be inconsistent.
        
        This helps identify data quality issues where H3 cells at different
        resolutions don't align properly.
        
        Returns:
            pandas.DataFrame: Customers with potentially conflicting resolution assignments
        """
        try:
            result = self.conn.execute("""
                WITH customer_resolutions AS (
                    SELECT 
                        customer_id, stock_point_id,
                        h3_resolution, cluster_id, h3_cell_id,
                        assignment_confidence, assignment_tier
                    FROM customer_stockpoint_cluster_assignment
                    WHERE status = 'ACTIVE'
                ),
                multi_resolution_customers AS (
                    SELECT 
                        customer_id, stock_point_id,
                        COUNT(DISTINCT h3_resolution) as resolution_count
                    FROM customer_resolutions
                    GROUP BY customer_id, stock_point_id
                    HAVING COUNT(DISTINCT h3_resolution) > 1
                )
                SELECT 
                    cr.customer_id, cr.stock_point_id,
                    cr.h3_resolution, cr.cluster_id, cr.h3_cell_id,
                    cr.assignment_confidence, cr.assignment_tier,
                    mrc.resolution_count
                FROM customer_resolutions cr
                INNER JOIN multi_resolution_customers mrc ON (
                    cr.customer_id = mrc.customer_id 
                    AND cr.stock_point_id = mrc.stock_point_id
                )
                ORDER BY cr.customer_id, cr.stock_point_id, cr.h3_resolution
            """).fetchall()
            
            if result:
                columns = [
                    'customer_id', 'stock_point_id', 'h3_resolution', 'cluster_id', 
                    'h3_cell_id', 'assignment_confidence', 'assignment_tier', 'resolution_count'
                ]
                conflicts_df = pd.DataFrame(result, columns=columns)
                unique_customers = conflicts_df['customer_id'].nunique()
                print(f"🔍 Found {unique_customers} customers with multiple resolution assignments")
                print("   (This may be normal if you intentionally use multiple resolutions)")
                return conflicts_df
            else:
                print("✅ No customers with multiple resolution assignments found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error checking resolution conflicts: {e}")
            return pd.DataFrame()

    def cleanup_old_records(self, days_to_keep: int = 365):
        """
        Archive or delete very old SUPERSEDED records to manage database size.
        
        Args:
            days_to_keep: Number of days of SUPERSEDED records to retain
            
        Returns:
            int: Number of records deleted
        """
        try:
            # Count records to be deleted
            count_result = self.conn.execute("""
                SELECT COUNT(*) 
                FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'SUPERSEDED' 
                  AND valid_to < CURRENT_DATE - INTERVAL ? DAYS
            """, [days_to_keep]).fetchone()[0]
            
            if count_result == 0:
                print(f"ℹ️ No SUPERSEDED records older than {days_to_keep} days found")
                return 0
            
            print(f"🗑️ Found {count_result} SUPERSEDED records older than {days_to_keep} days")
            
            # Delete old records
            self.conn.execute("""
                DELETE FROM customer_stockpoint_cluster_assignment 
                WHERE status = 'SUPERSEDED' 
                  AND valid_to < CURRENT_DATE - INTERVAL ? DAYS
            """, [days_to_keep])
            
            print(f"✅ Cleaned up {count_result} old SUPERSEDED records")
            return count_result
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            return 0

    def validate_h3_resolution_consistency(self, sample_size: int = 1000):
        """
        NEW METHOD: Validate that H3 cells at different resolutions are consistent
        for the same customer. This helps catch data quality issues.
        
        Args:
            sample_size: Number of customer-stock_point pairs to validate
            
        Returns:
            pandas.DataFrame: Any inconsistencies found
        """
        print(f"🔍 Validating H3 resolution consistency (sample size: {sample_size:,})...")
        
        try:
            # This would require H3 library to properly validate parent-child relationships
            # For now, we'll do a basic check for customers with multiple resolutions
            result = self.conn.execute(f"""
                WITH customer_multi_res AS (
                    SELECT 
                        customer_id, stock_point_id,
                        COUNT(DISTINCT h3_resolution) as resolution_count,
                        string_agg(
                            h3_resolution || ':' || COALESCE(h3_cell_id, 'NULL'), 
                            ', ' ORDER BY h3_resolution
                        ) as resolution_cells
                    FROM customer_stockpoint_cluster_assignment
                    WHERE status = 'ACTIVE'
                    GROUP BY customer_id, stock_point_id
                    HAVING COUNT(DISTINCT h3_resolution) > 1
                    LIMIT {sample_size}
                )
                SELECT * FROM customer_multi_res
                ORDER BY resolution_count DESC
            """).fetchall()
            
            if result:
                columns = ['customer_id', 'stock_point_id', 'resolution_count', 'resolution_cells']
                inconsistencies_df = pd.DataFrame(result, columns=columns)
                
                print(f"ℹ️ Found {len(inconsistencies_df)} customers with multiple H3 resolutions")
                print("   To fully validate consistency, you'll need H3 library to check parent-child relationships")
                
                return inconsistencies_df
            else:
                print("✅ No customers with multiple resolutions found in sample")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error validating H3 consistency: {e}")
            return pd.DataFrame()

    def get_change_summary(self, days_back: int = 7):
        """
        Get summary of changes in the last N days, including resolution changes.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            pandas.DataFrame: Summary of changes by date, reason, resolution, and status
        """
        try:
            result = self.conn.execute(f"""
                SELECT 
                    DATE(created_date) as change_date,
                    h3_resolution,
                    change_reason,
                    status,
                    changed_by,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(DISTINCT stock_point_id) as unique_stock_points,
                    AVG(assignment_confidence) as avg_confidence
                FROM customer_stockpoint_cluster_assignment
               WHERE created_date >= CURRENT_DATE - INTERVAL '{days_back} days'
                GROUP BY DATE(created_date), h3_resolution, change_reason, status, changed_by
                ORDER BY change_date DESC, h3_resolution, change_reason, status
            """).fetchall()
            
            if result:
                columns = [
                    'change_date', 'h3_resolution', 'change_reason', 'status', 'changed_by',
                    'record_count', 'unique_customers', 'unique_stock_points', 'avg_confidence'
                ]
                summary_df = pd.DataFrame(result, columns=columns)
                print(f"📈 Change summary for last {days_back} days (with H3 resolution breakdown):")
                return summary_df
            else:
                print(f"❌ No changes found in last {days_back} days")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error retrieving change summary: {e}")
            return pd.DataFrame()
        