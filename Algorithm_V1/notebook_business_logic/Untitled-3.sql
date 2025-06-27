
---------------------------------------------------------------------------------------
-- Clean up existing temp tables if they exist
DROP TABLE IF EXISTS #spCustomerScores;
DROP TABLE IF EXISTS #ScoreWeights;

-- Step 1: Create weight configuration table
CREATE TABLE #ScoreWeights (
    metric_name VARCHAR(50) PRIMARY KEY,
    weight DECIMAL(5,2),
    description VARCHAR(255)
);

-- Insert configurable weights (modify these values as needed)
INSERT INTO #ScoreWeights VALUES
    ('active_months_pct', 0.20, 'Percentage of active months in period'),
    ('avg_orders_per_active_month', 0.15, 'Order frequency per active month'),
    ('avg_revenue_per_month', 0.35, 'Revenue generation per month'),
    ('avg_qty_per_month', 0.15, 'Quantity purchased per month'),
    ('recency', 0.15, 'Days since last order (inverse)');

-- Step 2: Calculate precise 95th percentiles using ROW_NUMBER approach
WITH RankedOrders AS (
    SELECT
        StockPointID,
        avg_orders_per_active_month,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_orders_per_active_month) AS rn,
        COUNT(*) OVER (PARTITION BY StockPointID) AS cnt
    FROM #spCustAggregates
),
RankedQty AS (
    SELECT
        StockPointID,
        avg_qty_per_month,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_qty_per_month) AS rn,
        COUNT(*) OVER (PARTITION BY StockPointID) AS cnt
    FROM #spCustAggregates
),
RankedRevenue AS (
    SELECT
        StockPointID,
        avg_revenue_per_month,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_revenue_per_month) AS rn,
        COUNT(*) OVER (PARTITION BY StockPointID) AS cnt
    FROM #spCustAggregates
),
PercentileCalcs AS (
    SELECT
        o.StockPointID,
        MAX(CASE WHEN o.rn = CEILING(0.95 * o.cnt) THEN o.avg_orders_per_active_month END) AS orders_95p,
        MAX(CASE WHEN q.rn = CEILING(0.95 * q.cnt) THEN q.avg_qty_per_month END) AS qty_95p,
        MAX(CASE WHEN r.rn = CEILING(0.95 * r.cnt) THEN r.avg_revenue_per_month END) AS revenue_95p,
        MIN(o.cnt) AS customer_count
    FROM RankedOrders o
    JOIN RankedQty q ON o.StockPointID = q.StockPointID
    JOIN RankedRevenue r ON o.StockPointID = r.StockPointID
    GROUP BY o.StockPointID
    HAVING MIN(o.cnt) >= 10 -- Minimum customers required for percentile calculation
),

-- Step 3: Normalize metrics with robust edge case handling
NormalizedMetrics AS (
    SELECT
        c.StockPointID,
        c.CustomerID,
        
        -- Original metrics for reference
        c.active_months_pct,
        c.avg_orders_per_active_month,
        c.avg_qty_per_month,
        c.avg_revenue_per_month,
        c.days_since_last_order,
        
        -- Normalized metrics (4 decimal precision for calculations)
        CAST(ISNULL(c.active_months_pct, 0) AS DECIMAL(10,4)) AS norm_active_months,
        
        -- Normalized orders (capped at 95th percentile)
        CASE 
            WHEN p.orders_95p IS NULL THEN NULL -- Exclude StockPoints with insufficient data
            WHEN p.orders_95p <= 0 THEN 0
            WHEN c.avg_orders_per_active_month IS NULL THEN 0
            WHEN c.avg_orders_per_active_month <= p.orders_95p 
                THEN CAST(c.avg_orders_per_active_month / p.orders_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_orders,
        
        -- Normalized quantity (capped at 95th percentile)
        CASE 
            WHEN p.qty_95p IS NULL THEN NULL
            WHEN p.qty_95p <= 0 THEN 0
            WHEN c.avg_qty_per_month IS NULL THEN 0
            WHEN c.avg_qty_per_month <= p.qty_95p 
                THEN CAST(c.avg_qty_per_month / p.qty_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_qty,
        
        -- Normalized revenue (capped at 95th percentile)
        CASE 
            WHEN p.revenue_95p IS NULL THEN NULL
            WHEN p.revenue_95p <= 0 THEN 0
            WHEN c.avg_revenue_per_month IS NULL THEN 0
            WHEN c.avg_revenue_per_month <= p.revenue_95p 
                THEN CAST(c.avg_revenue_per_month / p.revenue_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_revenue,
        
        -- Recency score with 30-day half-life decay
        CAST( (1.0 / (1.0 + ISNULL(c.days_since_last_order, 999) / 30.0)) AS DECIMAL(10,4)) AS recency_score,
        
        -- Include customer count for reference
        p.customer_count
    FROM #spCustAggregates c
    LEFT JOIN PercentileCalcs p ON c.StockPointID = p.StockPointID
),

-- Step 4: Calculate composite scores and business tiers
FinalScores AS (
    SELECT
        StockPointID,
        CustomerID,
        
        -- Original metrics
        active_months_pct,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        days_since_last_order,
        
        -- Normalized components
        norm_active_months,
        norm_orders,
        norm_qty,
        norm_revenue,
        recency_score,
        customer_count,
        
        -- Composite score (0-100 scale)
        CASE 
            WHEN norm_orders IS NULL THEN NULL -- Exclude customers from StockPoints with insufficient data
            ELSE CAST(ROUND((
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') * recency_score
            ) * 100, 2) AS DECIMAL(5,2))
        END AS customer_score,
        
        -- Percentile rank within StockPoint
        CASE 
            WHEN norm_orders IS NULL THEN NULL
            ELSE PERCENT_RANK() OVER (
                PARTITION BY StockPointID 
                ORDER BY (
                    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months +
                    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders +
                    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue +
                    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty +
                    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') * recency_score
                )
            )
        END AS percentile_rank
    FROM NormalizedMetrics
)

-- Final output with business-friendly formatting
SELECT
    StockPointID,
    CustomerID,
    customer_score AS "CustomerScore",
    ROUND(percentile_rank * 100, 1) AS "PercentileRank",
    
    -- Business tier classification
    CASE 
        WHEN customer_score IS NULL THEN 'Insufficient Data'
        WHEN percentile_rank >= 0.9 THEN 'Platinum'
        WHEN percentile_rank >= 0.7 THEN 'Gold'
        WHEN percentile_rank >= 0.4 THEN 'Silver'
        ELSE 'Bronze'
    END AS "CustomerTier",
    
    -- Normalized components for diagnostics
    norm_active_months AS "NormActiveMonths",
    norm_orders AS "NormOrders",
    norm_revenue AS "NormRevenue",
    norm_qty AS "NormQuantity",
    recency_score AS "NormRecency",
    
    -- Original metrics for reference
    active_months_pct AS "ActiveMonthsPct",
    avg_orders_per_active_month AS "AvgOrdersPerMonth",
    avg_revenue_per_month AS "AvgRevenuePerMonth",
    days_since_last_order AS "DaysSinceLastOrder",
    
    -- Contextual information
    customer_count AS "StockPointCustomerCount"
INTO #spCustomerScores
FROM FinalScores
ORDER BY StockPointID, customer_score DESC;

-- Clean up temporary tables
DROP TABLE IF EXISTS #ScoreWeights;

-- Sample query to view results
SELECT TOP 100 * FROM #spCustomerScores;



--------------------------------------------------------------------------------------- G Method
DROP TABLE IF EXISTS #spCustScoresFinal;
DROP TABLE IF EXISTS #ScoreWeights;
-- Step 1: Create temporary configuration table to store weights
CREATE TABLE #ScoreWeights (
    metric_name VARCHAR(50) PRIMARY KEY,
    weight DECIMAL(5,2),
    description VARCHAR(255)
);

-- Insert configurable weights
INSERT INTO #ScoreWeights VALUES
    ('active_months_pct', 0.20, 'Percentage of active months in period'),
    ('avg_orders_per_active_month', 0.15, 'Order frequency per active month'),
    ('avg_revenue_per_month', 0.35, 'Revenue generation per month'),
    ('avg_qty_per_month', 0.15, 'Quantity purchased per month'),
    ('recency', 0.15, 'Days since last order (inverse)');

-- Validate weights sum to 1.0
IF (SELECT SUM(weight) FROM #ScoreWeights) != 1.0
    THROW 50001, 'Weights must sum to 1.0', 1;

-- Step 2: Assign row numbers for 95th percentile approximation
WITH RankedMetrics AS (
    SELECT
        StockPointID,
        CustomerID,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_orders_per_active_month) AS rn_orders,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_qty_per_month) AS rn_qty,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_revenue_per_month) AS rn_revenue,
        COUNT(*) OVER (PARTITION BY StockPointID) AS cnt
    FROM #spCustAggregates
),
GlobalRankedMetrics AS (
    SELECT
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        ROW_NUMBER() OVER (ORDER BY avg_orders_per_active_month) AS rn_orders,
        ROW_NUMBER() OVER (ORDER BY avg_qty_per_month) AS rn_qty,
        ROW_NUMBER() OVER (ORDER BY avg_revenue_per_month) AS rn_revenue,
        COUNT(*) OVER () AS cnt
    FROM #spCustAggregates
),
-- Compute 95th percentile per StockPointID and globally
PercentileCalcs AS (
    SELECT
        StockPointID,
        COALESCE(
            MAX(CASE WHEN rn_orders = CEILING(0.95 * cnt) THEN avg_orders_per_active_month END),
            (SELECT MAX(avg_orders_per_active_month) FROM GlobalRankedMetrics WHERE rn_orders = CEILING(0.95 * cnt))
        ) AS p95_orders,
        COALESCE(
            MAX(CASE WHEN rn_qty = CEILING(0.95 * cnt) THEN avg_qty_per_month END),
            (SELECT MAX(avg_qty_per_month) FROM GlobalRankedMetrics WHERE rn_qty = CEILING(0.95 * cnt))
        ) AS p95_qty,
        COALESCE(
            MAX(CASE WHEN rn_revenue = CEILING(0.95 * cnt) THEN avg_revenue_per_month END),
            (SELECT MAX(avg_revenue_per_month) FROM GlobalRankedMetrics WHERE rn_revenue = CEILING(0.95 * cnt))
        ) AS p95_revenue
    FROM RankedMetrics
    GROUP BY StockPointID
),
-- Step 3: Normalize metrics to 0-1 scale with null handling and offset
NormalizedMetrics AS (
    SELECT
        c.StockPointID,
        c.CustomerID,
        -- Raw metrics for optional output
        ISNULL(c.active_months_pct, 0) AS active_months_pct,
        ISNULL(c.avg_orders_per_active_month, 0) AS avg_orders_per_active_month,
        ISNULL(c.avg_qty_per_month, 0) AS avg_qty_per_month,
        ISNULL(c.avg_revenue_per_month, 0) AS avg_revenue_per_month,
        ISNULL(c.days_since_last_order, 9999) AS days_since_last_order,
        -- Normalized active_months_pct (already 0-1)
        CAST(ISNULL(c.active_months_pct, 0) AS DECIMAL(10,4)) AS norm_active_months_pct,
        -- Normalized orders (capped at 95th percentile with offset)
        CAST(
            CASE 
                WHEN p.p95_orders = 0 OR p.p95_orders IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_orders_per_active_month, 0) + 0.0001 < p.p95_orders 
                        THEN (ISNULL(c.avg_orders_per_active_month, 0) + 0.0001) 
                        ELSE p.p95_orders 
                END / p.p95_orders 
            END AS DECIMAL(10,4)
        ) AS norm_orders_per_active_month,
        -- Normalized quantity (capped at 95th percentile with offset)
        CAST(
            CASE 
                WHEN p.p95_qty = 0 OR p.p95_qty IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_qty_per_month, 0) + 0.0001 < p.p95_qty 
                        THEN (ISNULL(c.avg_qty_per_month, 0) + 0.0001) 
                        ELSE p.p95_qty 
                END / p.p95_qty 
            END AS DECIMAL(10,4)
        ) AS norm_qty_per_month,
        -- Normalized revenue (capped at 95th percentile)
        CAST(
            CASE 
                WHEN p.p95_revenue = 0 OR p.p95_revenue IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_revenue_per_month, 0) < p.p95_revenue 
                        THEN ISNULL(c.avg_revenue_per_month, 0) 
                        ELSE p.p95_revenue 
                END / p.p95_revenue 
            END AS DECIMAL(10,4)
        ) AS norm_revenue_per_month,
        -- Recency score with 30-day half-life decay
        CAST(
            1.0 / (1.0 + ISNULL(c.days_since_last_order, 9999) / 30.0)
            AS DECIMAL(10,4)
        ) AS norm_recency
    FROM #spCustAggregates c
    LEFT JOIN PercentileCalcs p ON c.StockPointID = p.StockPointID
),
-- Step 4: Compute composite score and optional percentile rank
FinalScores AS (
    SELECT
        StockPointID,
        CustomerID,
        -- Include raw and normalized metrics for debugging (comment out if not needed)
        active_months_pct,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        days_since_last_order,
        norm_active_months_pct,
        norm_orders_per_active_month,
        norm_qty_per_month,
        norm_revenue_per_month,
        norm_recency,
        -- Composite score (0-1 scale, rounded to 4 decimals)
        CAST(
            ROUND(
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months_pct +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders_per_active_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue_per_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty_per_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') * norm_recency,
                4
            ) AS DECIMAL(10,4)
        ) AS composite_customer_score,
        -- Optional percentile rank within each StockPointID (comment out if not needed)
        PERCENT_RANK() OVER (
            PARTITION BY StockPointID
            ORDER BY (
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months_pct +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders_per_active_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue_per_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty_per_month +
                (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') * norm_recency
            )
        ) AS percentile_rank
    FROM NormalizedMetrics
)
-- Final output (modify to 0-100 scale by uncommenting the alternative select)
SELECT
    StockPointID,
    CustomerID,
    composite_customer_score * 100 AS composite_customer_score, 
    CAST(percentile_rank AS DECIMAL(5,4)) AS percentile_rank,
    -- Include additional columns if needed (uncomment as required)
    active_months_pct,
    avg_orders_per_active_month,
    avg_qty_per_month,
    avg_revenue_per_month,
    days_since_last_order,
    norm_active_months_pct,
    norm_orders_per_active_month,
    norm_qty_per_month,
    norm_revenue_per_month,
    norm_recency
    -- For 0-100 scale, replace composite_customer_score with:
    -- CAST(composite_customer_score * 100 AS DECIMAL(5,2)) AS customer_score
INTO #spCustScoresFinal
FROM FinalScores
ORDER BY StockPointID, composite_customer_score DESC;

-- Clean up temporary table
DROP TABLE IF EXISTS #ScoreWeights;


SELECT 
	TOP 10 *
	--MIN(composite_customer_score * 100) Minn, 
	--MAX(composite_customer_score  * 100) Max
FROM #spCustScoresFinal


SELECT * 
INTO poc_stockpoint_customer_score
FROM #spCustScoresFinal















-----------------------------------------------------------------------------------
---- SCORING ALGO 1
DROP TABLE IF EXISTS #spCustScores;
DROP TABLE IF EXISTS #ScoreWeights;
-- Step 1: Create temporary configuration table to store weights
CREATE TABLE #ScoreWeights (
    metric_name VARCHAR(50) PRIMARY KEY,
    weight DECIMAL(5,2),
    description VARCHAR(255)
);

-- Insert configurable weights
INSERT INTO #ScoreWeights VALUES
    ('active_months_pct', 0.20, 'Percentage of active months in period'),
    ('avg_orders_per_active_month', 0.15, 'Order frequency per active month'),
    ('avg_revenue_per_month', 0.35, 'Revenue generation per month'),
    ('avg_qty_per_month', 0.15, 'Quantity purchased per month'),
    ('recency', 0.15, 'Days since last order (inverse)');
	 

---- Step 2: Assign row numbers for 95th percentile approximation per StockPointID
WITH RankedMetrics AS (
    SELECT
        StockPointID,
        CustomerID,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_orders_per_active_month) AS rn_orders,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_qty_per_month) AS rn_qty,
        ROW_NUMBER() OVER (PARTITION BY StockPointID ORDER BY avg_revenue_per_month) AS rn_revenue,
        COUNT(*) OVER (PARTITION BY StockPointID) AS cnt
    FROM #spCustAggregates
),
-- Compute 95th percentile for each metric per StockPointID
PercentileCalcs AS (
    SELECT
        StockPointID,
        MAX(CASE WHEN rn_orders = CEILING(0.95 * cnt) THEN avg_orders_per_active_month END) AS p95_orders,
        MAX(CASE WHEN rn_qty = CEILING(0.95 * cnt) THEN avg_qty_per_month END) AS p95_qty,
        MAX(CASE WHEN rn_revenue = CEILING(0.95 * cnt) THEN avg_revenue_per_month END) AS p95_revenue
    FROM RankedMetrics
    GROUP BY StockPointID
),
-- Step 3: Normalize metrics to 0-1 scale with null handling
NormalizedMetrics AS (
    SELECT
        c.StockPointID,
        c.CustomerID,
        -- Raw metrics for optional output
        ISNULL(c.active_months_pct, 0) AS active_months_pct,
        ISNULL(c.avg_orders_per_active_month, 0) AS avg_orders_per_active_month,
        ISNULL(c.avg_qty_per_month, 0) AS avg_qty_per_month,
        ISNULL(c.avg_revenue_per_month, 0) AS avg_revenue_per_month,
        ISNULL(c.days_since_last_order, 9999) AS days_since_last_order,
        -- Normalized active_months_pct (already 0-1)
        CAST(ISNULL(c.active_months_pct, 0) AS DECIMAL(10,4)) AS norm_active_months_pct,
        -- Normalized orders (capped at 95th percentile)
        CAST(
            CASE 
                WHEN p.p95_orders = 0 OR p.p95_orders IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_orders_per_active_month, 0) < p.p95_orders THEN ISNULL(c.avg_orders_per_active_month, 0)
                    ELSE p.p95_orders 
                END / p.p95_orders 
            END AS DECIMAL(10,4)
        ) AS norm_orders_per_active_month,
        -- Normalized quantity (capped at 95th percentile)
        CAST(
            CASE 
                WHEN p.p95_qty = 0 OR p.p95_qty IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_qty_per_month, 0) < p.p95_qty THEN ISNULL(c.avg_qty_per_month, 0)
                    ELSE p.p95_qty 
                END / p.p95_qty 
            END AS DECIMAL(10,4)
        ) AS norm_qty_per_month,
        -- Normalized revenue (capped at 95th percentile)
        CAST(
            CASE 
                WHEN p.p95_revenue = 0 OR p.p95_revenue IS NULL THEN 0 
                ELSE CASE 
                    WHEN ISNULL(c.avg_revenue_per_month, 0) < p.p95_revenue THEN ISNULL(c.avg_revenue_per_month, 0)
                    ELSE p.p95_revenue 
                END / p.p95_revenue 
            END AS DECIMAL(10,4)
        ) AS norm_revenue_per_month,
        -- Recency score with 30-day half-life decay
        CAST(
            1.0 / (1.0 + ISNULL(c.days_since_last_order, 9999) / 30.0)
            AS DECIMAL(10,4)
        ) AS norm_recency
    FROM #spCustAggregates c
    LEFT JOIN PercentileCalcs p ON c.StockPointID = p.StockPointID
),
-- Step 4: Compute composite score and optional percentile rank
FinalScores AS (
    SELECT
        StockPointID,
        CustomerID,
        -- Include raw and normalized metrics for debugging (comment out if not needed)
        active_months_pct,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        days_since_last_order,
        norm_active_months_pct,
        norm_orders_per_active_month,
        norm_qty_per_month,
        norm_revenue_per_month,
        norm_recency,
        -- Composite score (0-1 scale, rounded to 4 decimals)
        CAST(
            ROUND(
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months_pct +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders_per_active_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue_per_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty_per_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'recency') * norm_recency,
                4
            ) AS DECIMAL(10,4)
        ) AS composite_customer_score,
        -- Optional percentile rank within each StockPointID (comment out if not needed)
        PERCENT_RANK() OVER (
            PARTITION BY StockPointID
            ORDER BY (
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * norm_active_months_pct +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * norm_orders_per_active_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * norm_revenue_per_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * norm_qty_per_month +
                (SELECT Weight FROM #ScoreWeights WHERE metric_name = 'recency') * norm_recency
            )
        ) AS percentile_rank
    FROM NormalizedMetrics
)
-- Final output (modify to 0-100 scale by uncommenting the alternative select)
SELECT
    StockPointID,
    CustomerID,
	-- For 0-100 scale, replace composite_customer_score with:
    CAST(composite_customer_score * 100 AS DECIMAL(5,2)) AS customer_score,
    CAST(percentile_rank AS DECIMAL(5, 4)) percentile_rank,
     --Include additional columns if needed (uncomment as required)
     active_months_pct,
     avg_orders_per_active_month,
     avg_qty_per_month,
     avg_revenue_per_month,
     days_since_last_order,
     norm_active_months_pct,
     norm_orders_per_active_month,
     norm_qty_per_month,
     norm_revenue_per_month,
     norm_recency
INTO #spCustScores
FROM FinalScores
ORDER BY StockPointID, composite_customer_score DESC;

-- Clean up temporary table
DROP TABLE IF EXISTS #ScoreWeights;


----------------------------------------------------------------------------------------------
---- SCORING ALGO 2
DROP TABLE IF EXISTS #spCustScores2;
DROP TABLE IF EXISTS #ScoreWeights;
-- Step 1: Create weight configuration table
CREATE TABLE #ScoreWeights (
    metric_name VARCHAR(50) PRIMARY KEY,
    weight DECIMAL(5,2),
    description VARCHAR(255)
); 

-- Insert configurable weights
INSERT INTO #ScoreWeights VALUES
    ('active_months_pct', 0.20, 'Percentage of active months in period'),
    ('avg_orders_per_active_month', 0.15, 'Order frequency per active month'),
    ('avg_revenue_per_month', 0.35, 'Revenue generation per month'),
    ('avg_qty_per_month', 0.15, 'Quantity purchased per month'),
    ('recency', 0.15, 'Days since last order (inverse)');

-- Step 2: Calculate approximate 95th percentiles using window functions
WITH OrderedMetrics AS (
    SELECT
        StockPointID,
        avg_orders_per_active_month,
        avg_qty_per_month,
        avg_revenue_per_month,
        PERCENT_RANK() OVER (PARTITION BY StockPointID ORDER BY avg_orders_per_active_month) AS orders_pct_rank,
        PERCENT_RANK() OVER (PARTITION BY StockPointID ORDER BY avg_qty_per_month) AS qty_pct_rank,
        PERCENT_RANK() OVER (PARTITION BY StockPointID ORDER BY avg_revenue_per_month) AS revenue_pct_rank
    FROM #spCustAggregates
),
PercentileCalcs AS (
    SELECT
        StockPointID,
        MAX(CASE WHEN orders_pct_rank >= 0.95 THEN avg_orders_per_active_month END) AS orders_95p,
        MAX(CASE WHEN qty_pct_rank >= 0.95 THEN avg_qty_per_month END) AS qty_95p,
        MAX(CASE WHEN revenue_pct_rank >= 0.95 THEN avg_revenue_per_month END) AS revenue_95p,
        COUNT(*) AS customer_count
    FROM OrderedMetrics
    GROUP BY StockPointID
),

-- Step 3: Normalize metrics with SQL Server-compatible syntax
NormalizedMetrics AS (
    SELECT
        c.StockPointID,
        c.CustomerID,
        c.active_months_pct,
        c.avg_orders_per_active_month,
        c.avg_qty_per_month,
        c.avg_revenue_per_month,
        c.days_since_last_order,
        
        -- Normalized components
        CAST(ISNULL(c.active_months_pct, 0) AS DECIMAL(10,4)) AS norm_active_months,
        
        -- Replace LEAST with CASE statement
        CASE 
            WHEN p.orders_95p <= 0 OR p.customer_count < 20 THEN 0
            WHEN c.avg_orders_per_active_month IS NULL THEN 0
            WHEN c.avg_orders_per_active_month <= p.orders_95p 
                THEN CAST(c.avg_orders_per_active_month / p.orders_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_orders,
        
        CASE 
            WHEN p.qty_95p <= 0 OR p.customer_count < 20 THEN 0
            WHEN c.avg_qty_per_month IS NULL THEN 0
            WHEN c.avg_qty_per_month <= p.qty_95p 
                THEN CAST(c.avg_qty_per_month / p.qty_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_qty,
        
        CASE 
            WHEN p.revenue_95p <= 0 OR p.customer_count < 20 THEN 0
            WHEN c.avg_revenue_per_month IS NULL THEN 0
            WHEN c.avg_revenue_per_month <= p.revenue_95p 
                THEN CAST(c.avg_revenue_per_month / p.revenue_95p AS DECIMAL(10,4))
            ELSE 1.0
        END AS norm_revenue,
        
        CAST(1.0 / (1.0 + ISNULL(c.days_since_last_order, 999) / 30.0) AS DECIMAL(10,4)) AS recency_score
    FROM #spCustAggregates c
    LEFT JOIN PercentileCalcs p ON c.StockPointID = p.StockPointID
)

-- Step 4: Calculate final scores using configurable weights
SELECT
    n.StockPointID,
    n.CustomerID,
    CAST(ROUND((
        (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') * n.norm_active_months +
        (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') * n.norm_orders +
        (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') * n.norm_revenue +
        (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') * n.norm_qty +
        (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') * n.recency_score
    ) * 100, 2) AS DECIMAL(5,2)) AS customer_score,
    
    -- Show normalized values and weights for transparency
    n.norm_active_months,
    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'active_months_pct') AS weight_active_months,
    
    n.norm_orders,
    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_orders_per_active_month') AS weight_orders,
    
    n.norm_revenue,
    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_revenue_per_month') AS weight_revenue,
    
    n.norm_qty,
    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'avg_qty_per_month') AS weight_quantity,
    
    n.recency_score,
    (SELECT weight FROM #ScoreWeights WHERE metric_name = 'recency') AS weight_recency
INTO #spCustScores2
FROM NormalizedMetrics n
ORDER BY n.StockPointID, customer_score DESC; 


DROP TABLE IF EXISTS #ScoreWeights;

SELECT * FROM #spCustScores ORDER BY StockPointID, customer_score DESC
SELECT * FROM #spCustScores2 ORDER BY StockPointID, customer_score DESC










SELECT TOP 1 * FROM tblorderSales WITH (NOLOCK) 
WHERE Central_BusinessID = 76 AND orderstatusID NOT IN (215, 216) -- NOT CANCELLED OR RTO

SELECT MIN(CreatedDate) FROM tblorderSales WITH (NOLOCK) 
WHERE Central_BusinessID = 76







SELECT DATEADD(MONTH, -3, CAST(GETDATE() AS DATE))
DROP TABLE IF EXISTS #bt_cust;
SELECT  
	CustomerID,  
	CAST(COUNT(DISTINCT MONTH(Createddate)) / 6.0 AS DECIMAL(5,2)) active_months_pct,	
	COUNT(DISTINCT OrderID) / (COUNT(DISTINCT MONTH(Createddate))) avg_orders_per_active_month, 
	SUM(Quantity) / (COUNT(DISTINCT MONTH(Createddate))) avg_qty_per_month,
	count(DISTINCT MONTH(Createddate)) active_month,
	COUNT(DISTINCT OrderID) n_orders, 
	SUM(Quantity) total_qty,
	MIN(Createddate) first_order_date,
	MAX(Createddate) last_order_date,
	DATEDIFF(DAY, MAX(Createddate), CAST(GETDATE() AS DATE)) AS days_since_last_order
INTO #bt_cust
FROM tblorderSales WITH (NOLOCK)
--FROM tblManuDashSales WITH (NOLOCK)
WHERE 
Central_BusinessID = 76
AND YEAR(Createddate) = 2025
AND orderstatusID <> 215
--AND YEAR(deliverydate) = 2025
AND CustomerID IN (
		1864803,1864803, 1878283, 2087855, 2502546,
		2669508, 3043964, 3899753, 3906037,
		4331522, 4388539,4434598,4518095,
		5258797,5261052,5274465,5295933,
		5316206,5334167,5344647)
GROUP BY CustomerID


; WITH CTE AS (
	SELECT *,
	(
	 (30 * (1 - (days_since_last_order / 180)))  --- Recency Score (30% weight) 
	 + (active_months_pct * avg_orders_per_active_month * 25) ---- Frequency Score (25% weight)
	 + ((total_qty * 12.5) + (avg_qty_per_month*12.5)) ---- Volume Score (25% weight)
	 + (active_months_pct * 20)--- Consistency Score (20% weight)
	 ) / 100.0 AS SCORE
	FROM #bt_cust 
)
SELECT * FROM CTE ORDER BY SCORE DESC 

;

SELECT DATEDIFF(DAY, '2022-09-01', '2022-09-02');
/*
30 * (1 - (days_since_last_order / 180))
customer_data['volume_score'] = (
    customer_data['total_quantity'] * 0.125 +  # 10% weight
    customer_data['avg_quantity_per_active_month'] * 0.125  # 15% weight
)


*/




SELECT