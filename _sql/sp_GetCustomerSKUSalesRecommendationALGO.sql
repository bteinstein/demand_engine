USE VconnectMasterDWR;
GO

 
CREATE OR ALTER PROCEDURE [dbo].[sp_GetCustomerSKUSalesRecommendationALGO] 

/*
EXEC [dbo].[sp_GetCustomerSKUSalesRecommendationALGO] 
*/

 

 exec [getCustomerSKURecommendationChecks]

AS
BEGIN
    SET NOCOUNT ON;
    SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED; -- For speed, similar to NOLOCK

	----------------------------------------------------------------------------------------------------------------------------------
	-- ACTIVE RECOMMENDATION
	DROP TABLE IF EXISTS #rec_active;
	SELECT * INTO #rec_active 
	FROM CustomerSKUSalesRecommendationLog WITH (NOLOCK)
	WHERE [status] = 'Active' AND ISNULL(InventoryCheck,'') <>  'No' ; --AND FCID = 1647113 

	
	SELECT TOP 10 * FROM #rec_active;
	----------------------------------------------------------------------------------------------------------------------------------
	--- SELECTED SP 
	DROP TABLE IF EXISTS #rec_active_modified;
	SELECT 
		A.*,
		B.sp_tag as spSKUtag,
		CASE 
			WHEN B.sp_tag = 'Express' THEN 50 
			WHEN B.sp_tag = 'Core' THEN 30 
			WHEN B.sp_tag = 'Standard' THEN 15 
			WHEN B.sp_tag = 'Standard-Inactive' THEN 5
			ELSE 0
		END AS spSKUtagWeight,
		CASE 
			WHEN A.Medium = 'RF' THEN 45 
			WHEN A.Medium = 'Once in 8 Weeks' THEN 25 
			WHEN A.Medium = 'Past Purchased' THEN 20 
			WHEN A.Medium = 'Probability' THEN 10
			WHEN A.Medium = 'Never Purchased' THEN 5
			ELSE 0
		END AS MediumWeight,  
		CAST (
			CASE  
				WHEN MAX(ISNULL(EstimatedQuantity, 0)) OVER (PARTITION BY CustomerID) 
					 = MIN(ISNULL(EstimatedQuantity, 0)) OVER (PARTITION BY CustomerID)
				THEN 1
				ELSE 1 + (ISNULL(EstimatedQuantity, 0) - MIN(ISNULL(EstimatedQuantity, 0)) OVER (PARTITION BY CustomerID)) * 99.0 /
						 NULLIF(MAX(ISNULL(EstimatedQuantity, 0)) OVER (PARTITION BY CustomerID) - 
								MIN(ISNULL(EstimatedQuantity, 0)) OVER (PARTITION BY CustomerID), 0)
			END  
		 AS DECIMAL(5, 2) ) AS EstimatedQuantityScaled,
		CAST(0 AS DECIMAL(5, 2)) AS CustomerSKUscore 
		,CAST(0 AS DECIMAL(5, 2)) AS CustomerSKUscoreRank 
		,CAST(0 AS DECIMAL(5, 2)) AS CustomerSKUscoreStandardize
	INTO #rec_active_modified 
	FROM #rec_active A 
	INNER JOIN poc_sku_tagging_sp B ON B.fc_id = A.FCID AND B.sku_id = A.SkuId
	WHERE ISNULL(EstimatedQuantity,0) > 0 ; ---

	----------------------------------------------------------------------------------------------------------------------------------
	-- UPDATES
	---------------- CUSTOMER SKU SCORING AND RANKING 
	UPDATE #rec_active_modified
	SET EstimatedQuantity = ISNULL(EstimatedQuantity,0);

	UPDATE #rec_active_modified
	SET CustomerSKUscore = CAST(
		100 * POWER(spSKUtagWeight / 100.0, 1.0) * 
			  POWER(MediumWeight / 100.0, 1.0) * 
			  POWER(EstimatedQuantityScaled / 100.0, 1.0)
		AS DECIMAL(5, 2));
	 
	WITH StandardizedScores AS (
		SELECT
			CustomerID,
			SKUID,
			CustomerSKUscore,
			CAST (
				CASE  
					WHEN MAX(ISNULL(CustomerSKUscore, 0)) OVER (PARTITION BY CustomerID) 
						 = MIN(ISNULL(CustomerSKUscore, 0)) OVER (PARTITION BY CustomerID)
					THEN 1
					ELSE 1 + (ISNULL(CustomerSKUscore, 0) - MIN(ISNULL(CustomerSKUscore, 0)) OVER (PARTITION BY CustomerID)) * 99.0 /
							 NULLIF(MAX(ISNULL(CustomerSKUscore, 0)) OVER (PARTITION BY CustomerID) - 
									MIN(ISNULL(CustomerSKUscore, 0)) OVER (PARTITION BY CustomerID), 0)
				END  
			AS DECIMAL(5, 2) ) AS CustomerSKUscoreStandardize 
		FROM #rec_active_modified
	)
	-- Then update using the CTE
	UPDATE rac
	SET CustomerSKUscoreStandardize = ss.CustomerSKUscoreStandardize
	FROM #rec_active_modified rac
	INNER JOIN StandardizedScores ss ON rac.CustomerID = ss.CustomerID AND rac.SKUID = ss.SKUID;

	WITH RankedScores AS (
		SELECT
			CustomerID,
			SKUID,
			CustomerSKUscore,
			RANK() OVER (PARTITION BY CustomerID ORDER BY CustomerSKUscore DESC) AS CustomerSKUscoreRank, 
			RANK() OVER (PARTITION BY CustomerID ORDER BY CustomerSKUscoreStandardize DESC) AS CustomerSKUscoreStandardize
		FROM #rec_active_modified
	)
	-- Step 2: Update the original table using the CTE
	UPDATE rac
	SET 
		rac.CustomerSKUscoreRank = CAST(rs.CustomerSKUscoreRank AS INT)
	FROM #rec_active_modified rac
	INNER JOIN RankedScores rs ON rac.CustomerID = rs.CustomerID AND rac.SKUID = rs.SKUID;

	----------------------------------------------------------------------------------------------------
	-- SP - CUSTOMER - SKU
	DROP TABLE IF EXISTS #rec_active_modified_score;
	SELECT  
		FCID,
		CustomerId,	 
		SKUID,	
		ProductName, 
		Output,		
		LastDeliveredDate,
		InventoryCheck,	 
		spSKUtag AS ProductTag, 
		Medium, 
		EstimatedQuantity, 
		CustomerSKUscore,
		CustomerSKUscoreStandardize,
		CAST(CustomerSKUscoreRank AS INT) CustomerSKUscoreRank
	INTO #rec_active_modified_score
	FROM #rec_active_modified   
	ORDER BY CustomerID, CAST(CustomerSKUscoreRank AS INT)
 

	----------------------------------------------------------------------------------------------------
	-- AFFINITY SCORE
		DROP TABLE IF EXISTS #CustomerAffinityScore;
	-- Step 1: Calculate raw customer affinity scores
	WITH DIM_Customer AS (
		SELECT DISTINCT 
			FCID, 
			CustomerID,	
			Name, 
			StateName,	
			Region,	 
			TRY_CAST(Latitude AS FLOAT) Latitude ,	
			TRY_CAST(Longitude AS FLOAT) Longitude, 
			LGA,	
			LCDA	
			FROM #rec_active_modified
		), 
	CustomerAffinity AS (
		SELECT 
			FCID, 
			CustomerID,
			COUNT(*) as TotalSKUs,
			AVG(CustomerSKUscore) as AvgSKUScore,
			SUM(CustomerSKUscore) as TotalSKUScore,
			MAX(CustomerSKUscore) as BestSKUScore,
			SUM(EstimatedQuantity) as TotalEstimatedVolume,
			SUM(CASE WHEN Medium = 'RF' THEN 1 ELSE 0 END) as RFcount,
        
			-- High-value SKU metrics
			SUM(CASE WHEN spSKUtag IN ('Express','Core') THEN 1 ELSE 0 END) as HighValueSKUs,
			AVG(CASE WHEN spSKUtag IN ('Express','Core') THEN CustomerSKUscore ELSE NULL END) as HighValueAvgScore,
			SUM(CASE WHEN spSKUtag IN ('Express','Core') THEN CustomerSKUscore ELSE 0 END) as HighValueTotalScore,
			SUM(CASE WHEN spSKUtag IN ('Express','Core') THEN EstimatedQuantity ELSE 0 END) as HighValueEstimatedVolume,
        
			-- Individual SKU tag counts
			SUM(CASE WHEN spSKUtag = 'Express' THEN 1 ELSE 0 END) as ExpressSKUs,
			SUM(CASE WHEN spSKUtag = 'Core' THEN 1 ELSE 0 END) as CoreSKUs,
        
			-- Raw affinity score (updated to include high-value focus)
			CAST(
				0.3 * AVG(CustomerSKUscore) +                              -- Portfolio quality
				0.2 * (SUM(EstimatedQuantity)/1000.0) +                      -- Volume potential  
				0.2 * (SUM(CASE WHEN Medium = 'RF' THEN 1 ELSE 0 END) * 20) + -- Engagement level
				0.2 * ISNULL(AVG(CASE WHEN spSKUtag IN ('Express','Core') THEN CustomerSKUscore ELSE NULL END), 0) + -- High-value SKU quality
				0.1 * (SUM(CASE WHEN spSKUtag IN ('Express','Core') THEN 1 ELSE 0 END) * 10) -- High-value SKU count
			AS DECIMAL(5,2)) as CustomerAffinityScore_Raw
		FROM #rec_active_modified
		GROUP BY FCID,  CustomerID 
	),
	-- Step 2: Standardize the affinity scores to 1-100 range
	StandardizedAffinity AS (
		SELECT 
			*,
			CAST (
				CASE  
					WHEN MAX(ISNULL(CustomerAffinityScore_Raw, 0)) OVER () 
						 = MIN(ISNULL(CustomerAffinityScore_Raw, 0)) OVER ()
					THEN 1
					ELSE 1 + (ISNULL(CustomerAffinityScore_Raw, 0) - MIN(ISNULL(CustomerAffinityScore_Raw, 0)) OVER ()) * 99.0 /
							 NULLIF(MAX(ISNULL(CustomerAffinityScore_Raw, 0)) OVER () - 
									MIN(ISNULL(CustomerAffinityScore_Raw, 0)) OVER (), 0)
				END  
			AS DECIMAL(5, 2) ) AS CustomerAffinityScore_Standardized,
			-- Add rank
			RANK() OVER (PARTITION BY FCID ORDER BY CustomerAffinityScore_Raw DESC) AS CustomerAffinityRank
		FROM CustomerAffinity
	)
	-- Step 3: Final output with all metrics
	SELECT 
		D.FCID,  
		D.CustomerID,	
		D.Name, 
		D.StateName,	
		D.Region,	 
		D.Latitude,	
		D.Longitude, 
		D.LGA,	
		D.LCDA,
		TotalSKUs,
		AvgSKUScore,
		TotalEstimatedVolume,
		RFcount,
    
		-- High-value SKU metrics
		HighValueSKUs,
		HighValueAvgScore,
		HighValueTotalScore,
		HighValueEstimatedVolume,
		ExpressSKUs,
		CoreSKUs,
    
		CustomerAffinityScore_Raw,
		CustomerAffinityScore_Standardized,
		CustomerAffinityRank

	INTO #CustomerAffinityScore
	FROM DIM_Customer D
	LEFT JOIN StandardizedAffinity S ON S.CustomerID = D.CustomerID AND S.FCID = D.FCID
	ORDER BY CustomerAffinityRank;


	----------------------------------------------------------------------------------------------------
	-- SP DIM
	DROP TABLE IF EXISTS #SP_DIM;
	SELECT 
		sp.Stock_Point_ID, sp.Stock_point_Name, 
		bm.lattitude, bm.longitude
	INTO #SP_DIM
	FROM Stock_Point_Master sp WITH (NOLOCK)  
	INNER JOIN BusinessMaster bm WITH (NOLOCK) ON sp.Stock_Point_ID = bm.Contentid
	WHERE sp.Fulfilement_Center_ID = 76
		AND sp.Status = 1
		AND sp.Warehouse_Type IN (1, 3)
		AND sp.Stock_Point_Name NOT LIKE '%TEST%'
		AND sp.Is_Mfc = 1

	---------------------------------------------------------
	--- RESULT
	SELECT * FROM #rec_active_modified_score
	SELECT * FROM #CustomerAffinityScore
	SELECT * FROM #SP_DIM
	
	--SELECT TOP 10 * FROM #rec_active_modified_score WHERE CustomerID IN (1680756, 2060) ORDER BY CustomerID, CustomerSKUscoreRank  
	--SELECT TOP 10 * FROM #CustomerAffinityScore WHERE CustomerID IN (1680756, 2060)

	--SELECT FCID, COUNT(DISTINCT CUSTOMERID) NC, MIN(CustomerAffinityRanK) MINR, MAX(CustomerAffinityRanK) MAXR FROM #CustomerAffinityScore GROUP BY FCID ORDER BY FCID  --- 1038


END

GO
   