USE VconnectMasterDWR;

------------------ BASE TABLE Customer Created Orders 
----- 3 Months ---> Date Size: 2,627,683 || ETA: 48 Secs
----- 6 Months ---> Date Size: 3,148,422 || ETA: 2mins Secs
DROP TABLE IF EXISTS #OrderCreated;
SELECT --- TOP 10
	CreatedDate, DeliveredDate, BusinessID as Stock_Point_ID, CustomerID, -- Stock_point_Name, 
	orderid, itemid,  SKUCode, Quantity, Price, Category_ID, 
	OrderStatusID, OrderStatus, CityID,  townid, mode1, IsSelfPickUP --- Category_Name, 
INTO #OrderCreated
FROM tblorderSales WITH (NOLOCK) 
WHERE Central_BusinessID = 76 
	  AND orderstatusID NOT IN (215, 216) -- NOT CANCELLED OR RTO
	  AND CAST(CreatedDate AS DATE) >= DATEADD(MONTH, -6, CAST(GETDATE() AS DATE))


--Select distinct orderstatusid from tblorderSales WHERE Central_BusinessID = 76 AND CAST(CreatedDate AS DATE) >= DATEADD(MONTH, -6, CAST(GETDATE() AS DATE))
--Select top 100 * from #OrderCreated where OrderStatusID IN (210,212,213)

--210
--212
--213
--214
--215
--216
--225



SELECT TOP 10 
	--MAX(CreatedDate, DeliveredDate) AS MAX_DATE, 
	*
FROM #OrderCreated
WHERE DeliveredDate IS NULL


--SELECT EOMONTH(DATEADD(MONTH, -6, CAST(GETDATE() AS DATE)))


	   

------------------ 2. Compute Customer Aggregates 
--- LEVEL STOCKPOINT AND Customer
--- ETA: 8 Secs
DROP TABLE IF EXISTS #spCustAggregates;
SELECT   
	Stock_Point_ID as StockPointID, CustomerID,  
	CAST(COUNT(DISTINCT MONTH(Createddate)) / 6.0 AS DECIMAL(5,2)) active_months_pct,	
	COUNT(DISTINCT OrderID) / (COUNT(DISTINCT MONTH(Createddate))) avg_orders_per_active_month, 
	SUM(Quantity) / (COUNT(DISTINCT MONTH(CreatedDate))) avg_qty_per_month,
	SUM(Price) / (COUNT(DISTINCT MONTH(CreatedDate))) avg_revenue_per_month,
	count(DISTINCT MONTH(CreatedDate)) active_month,
	COUNT(DISTINCT OrderID) n_orders, 
	SUM(Quantity) total_qty,
	SUM(Price) total_revenue,
	MIN(Createddate) first_order_date,
	CASE  
		WHEN MAX(DeliveredDate) > MAX(Createddate) 
		THEN MAX(DeliveredDate)
		ELSE MAX(Createddate) last_order_date,
	CASE  
		WHEN MAX(Createddate) < MAX(DeliveredDate) 
		THEN DATEDIFF(DAY, MAX(Createddate), CAST(GETDATE() AS DATE))  
	ELSE DATEDIFF(DAY, MAX(DeliveredDate), CAST(GETDATE() AS DATE)) 
	END AS days_since_last_order
INTO #spCustAggregates
FROM #OrderCreated 
GROUP BY Stock_Point_ID, CustomerID


--WHERE   CustomerID IN (
--		1864803,1864803, 1878283, 2087855, 2502546,
--		2669508, 3043964, 3899753, 3906037,
--		4331522, 4388539,4434598,4518095,
--		5258797,5261052,5274465,5295933,
--		5316206,5334167,5344647)

SELECT COUNT(*) FROM #spCustAggregates
SELECT TOP 10 * FROM #spCustAggregates

------------- 3. CUSTOMER SCORING -----------------
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

-- Step 2: Compute 95th percentile per StockPointID with global fallback
WITH PercentileCalcs AS (
    SELECT DISTINCT
        StockPointID,
        -- Use PERCENTILE_CONT for accurate 95th percentile calculation
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_orders_per_active_month) 
            OVER (PARTITION BY StockPointID) AS p95_orders,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_qty_per_month) 
            OVER (PARTITION BY StockPointID) AS p95_qty,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_revenue_per_month) 
            OVER (PARTITION BY StockPointID) AS p95_revenue,
        -- Also calculate global percentiles as fallback
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_orders_per_active_month) 
            OVER () AS global_p95_orders,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_qty_per_month) 
            OVER () AS global_p95_qty,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY avg_revenue_per_month) 
            OVER () AS global_p95_revenue
    FROM #spCustAggregates
),
PercentileCalcsFinal AS (
    SELECT 
        StockPointID,
        -- Use global percentile if StockPointID has too few records (optional logic)
        CASE WHEN p95_orders IS NULL OR p95_orders = 0 THEN global_p95_orders ELSE p95_orders END AS p95_orders,
        CASE WHEN p95_qty IS NULL OR p95_qty = 0 THEN global_p95_qty ELSE p95_qty END AS p95_qty,
        CASE WHEN p95_revenue IS NULL OR p95_revenue = 0 THEN global_p95_revenue ELSE p95_revenue END AS p95_revenue
    FROM PercentileCalcs
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
    LEFT JOIN PercentileCalcsFinal p ON c.StockPointID = p.StockPointID
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
-- Final output - FIXED: Changed DECIMAL(5,4) to DECIMAL(7,4) to accommodate 0-100 scale
SELECT
    StockPointID,
    CustomerID,
    CAST(composite_customer_score * 100 AS DECIMAL(7,4)) AS composite_customer_score, 
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
    -- CAST(composite_customer_score * 100 AS DECIMAL(7,2)) AS customer_score
INTO #spCustScoresFinal
FROM FinalScores
ORDER BY StockPointID, composite_customer_score DESC;

-- Clean up temporary table
DROP TABLE IF EXISTS #ScoreWeights;


-----
DROP TABLE IF EXISTS poc_stockpoint_customer_score;
SELECT 
	*, CAST(GETDATE() AS DATE) AS UpdateDate
INTO poc_stockpoint_customer_score
FROM #spCustScoresFinal

SELECT 
	TOP 1000 *	
	--MIN(composite_customer_score * 100) Minn, 
	--MAX(composite_customer_score  * 100) Max
FROM poc_stockpoint_customer_score ORDER BY StockPointID, composite_customer_score DESC
 

--SELECT max(days_since_last_order) FROM poc_stockpoint_customer_score

--select 182/30

SELECT TOP 10 * FROM #OrderCreated WHERE CustomerID = 4529739 ORDER BY CreatedDate DESC
SELECT TOP 3 * FROM tblorderSales WHERE CustomerID = 4529739 ORDER BY CreatedDate DESC
SELECT TOP 3 * FROM tblmanudashsales WHERE CustomerID = 4529739 and CENTRAL_BUSINESSID = 76 ORDER BY DeliveryDate DESC

eliveredDate
2025-01-10 09:25:31.897
