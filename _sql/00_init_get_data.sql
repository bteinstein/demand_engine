USE VconnectMasterDWR; 
DECLARE @LAST1YEAR AS DATE = CAST(DATEADD(DAY, -365, GETDATE()) AS DATE); 

IF OBJECT_ID('tempdb..#fact_sales_temp') IS NOT NULL DROP TABLE #fact_sales_temp;
SELECT TOP 100 
	*,
	CAST(deliverydate AS DATE) as deliverydate_
INTO #fact_sales_temp
FROM tblManuDashSales WITH (NOLOCK)
WHERE Central_businessid = 76

 ------------------------------------------------------------------------
-- 1. SKU DIM
--'SKUID', 'ManufacturerID', 'Manufacturer', 
--'CategoryID', 'Category', 'SegmentID', 'Segment'
-------------------------------------------------------------------------------- 
IF OBJECT_ID('tempdb..#dim_sku') IS NOT NULL DROP TABLE #dim_sku;
SELECT
    b.contentid as SKUID, 
	b.SKUCODE,
    b.ProductName,  
	b.category_id, c.Category_Name, 
    c.Segment, 
    b.brand_id, bb.brand_name AS Brand_Name,
    b.Manufacture_ID as Manufacturer_ID, m.Manufacture_Name as Manufacturer_Name
INTO #dim_sku
FROM BusinessStore b
INNER JOIN Brand_master bb ON b.brand_id = bb.brand_id
INNER JOIN category_master c ON b.category_id = c.category_id
INNER JOIN manufacture_master m ON m.Manufacture_ID = b.Manufacture_ID
WHERE Businessid = 76 

------------------------------------------------------------------------
-- 2. CUSTOMER DIM
--'CustomerID', 'State', 'City', 'Town', 'Recency', 'Frequency', 'Monetary'
-------------------------------------------------------------------------------- 
IF OBJECT_ID('tempdb..#customer_rfm') IS NOT NULL DROP TABLE #customer_rfm;
SELECT
    CustomerID,
    DATEDIFF(day, MAX(deliverydate), GETDATE()) AS Recency,
    COUNT(distinct deliverydate_) AS Frequency,
    SUM(Price) AS Monetary
INTO #customer_rfm
FROM #fact_sales_temp
GROUP BY CustomerID ;

---------------------------------------------------------------------------------------------------------------------------------------------------
---- Customer Location  
IF OBJECT_ID('tempdb..#dim_location') IS NOT NULL DROP TABLE #dim_location;
SELECT 
	t.Contentid as TownID,
	t.CityID,
	t.StateID StateID,
	t.CountryID CountryID,
	t.TownName, 
	c.CityName, 
	s.StateName,
	cc.CountryName 
INTO #dim_location
FROM Townmaster t with (nolock)  
INNER JOIN CityMaster c with (nolock) on t.CityID = c.ContentID
INNER JOIN StateMaster s with (nolock) on t.StateID = s.ContentID
INNER JOIN CountryMaster cc with (nolock) on t.CountryID = cc.ContentID
WHERE ISNULL(t.Status,0) = 1 and t.CountryID = 1

-- Create indexes on temp table for better performance
CREATE CLUSTERED INDEX IX_dim_location_TownID ON #dim_location (TownID);
CREATE NONCLUSTERED INDEX IX_dim_location_CityState ON #dim_location (CityName, StateName);


-- Step 2: Get latest records using window function (more efficient than correlated subquery) 
IF OBJECT_ID('tempdb..#latest_records') IS NOT NULL DROP TABLE #latest_records;
SELECT 
    BusinessID, CustomerID, ContactName, ContactPhone, 
    StateName, CityName, TownID, TownName, Latitude, 
    Longitude, CustomerTypeID, CustomerModeName, 
    CustomerType, CustomerStatus, ContentID
INTO #latest_records
FROM (
    SELECT 
        BusinessID, CustomerID, ContactName, ContactPhone, 
        StateName, CityName, TownID, TownName, Latitude, 
        Longitude, CustomerTypeID, CustomerModeName, 
        CustomerType, CustomerStatus, ContentID,
        ROW_NUMBER() OVER (PARTITION BY CustomerID ORDER BY ContentID DESC) AS rn
    FROM OmniDR..AddressBook WITH (NOLOCK)   
    WHERE ISNULL(Status, 0) IN (0, 1, 2, 7) 
      AND CustomerID IS NOT NULL
      AND BusinessID = 76 
) ranked
WHERE rn = 1;

-- Step 2: Add TownID_MOD to smaller dataset
ALTER TABLE #latest_records ADD TownID_MOD INT;

--- Update AddressBook
UPDATE lr
SET TownID_MOD = CASE 
    WHEN lr.TownID IS NOT NULL THEN lr.TownID 
    ELSE dl.TownID
END
FROM #latest_records lr
LEFT JOIN #dim_location dl ON dl.CityName = lr.CityName AND dl.StateName = lr.StateName;

-- Create index for next step
CREATE CLUSTERED INDEX IX_latest_records_CustomerID ON #latest_records (CustomerID);

IF OBJECT_ID('tempdb..#BusinessAddressbook') IS NOT NULL DROP TABLE #BusinessAddressbook;
SELECT  
    a.BusinessID,
    a.CustomerID,
    a.ContactName,
    a.ContactPhone,
    CASE WHEN ISNULL(l.StateID, 0) <> 0 THEN l.StateID END AS StateID,
    ISNULL(l.StateName, a.StateName) AS StateName,
    ISNULL(l.CityID, 0) AS CityID,
    ISNULL(l.CityName, a.CityName) AS CityName,
    CASE WHEN ISNULL(a.TownID_MOD, 0) <> 0 THEN a.TownID_MOD END AS TownID,
    ISNULL(l.TownName, a.TownName) AS TownName,
    a.Latitude,
    a.Longitude,
    a.CustomerTypeID,
    a.CustomerModeName,
    a.CustomerType,
    a.CustomerStatus
INTO #BusinessAddressbook 
FROM #latest_records a
LEFT JOIN #dim_location l ON l.TownID = a.TownID_MOD;

-- Clean up intermediate temp tables
--DROP TABLE #latest_records;
--DROP TABLE #BusinessAddressbook_temp;
--DROP TABLE #dim_location;
 
---------------- BASIC CHECK ----------------
--SELECT COUNT( *) FROM #BusinessAddressbook --- 340,970
--SELECT 
--	SUM(CASE WHEN TownID IS NULL THEN 1 ELSE 0 END) nullTownID, --- 88,202, 67,185  
--	SUM(CASE WHEN TownID IS NULL AND (ISNULL(StateName,'') <> '' OR ISNULL(CityName,'') <> '' OR  ISNULL(TownName,'') <> '' ) THEN 1 ELSE 0 END) nullTownID_wStateCityTown, --- 70,385 	46,255 
--	SUM(CASE WHEN TownID IS NULL AND (ISNULL(StateName,'') <> '' AND ISNULL(CityName,'') <> ''  ) THEN 1 ELSE 0 END) nullTownID_wStateCity, --- 61,400  37,169
--	COUNT(DISTINCT CustomerID) - SUM(CASE WHEN TownID IS NULL THEN 1 ELSE 0 END) number_with_town -- 252,768 273,851
--FROM #BusinessAddressbook a
 
--SELECT 
--	c.CustomerID, c.Recency, c.Frequency, c.Monetary,
--	b.StateName, b.CityName, b.TownID, b.TownName,
--	CASE WHEN b.TownID IS NULL THEN 1 ELSE 0 END AS MissingTown
--FROM #customer_rfm as c
--LEFT JOIN #BusinessAddressbook b ON c.CustomerID = b.CustomerID
--WHERE b.TownID IS NULL


--SELECT TOP 10 * FROM #BusinessAddressbook WHERE CustomerID = 5368079
--SELECT COUNT(*) FROM #BusinessAddressbook

 
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
-- Attempt to use Customers freq Town from Sales Table if the TownID is missing 
ALTER TABLE #BusinessAddressbook ADD MOD_LOCATION INT;
-- Update missing location data in BusinessAddressbook using order history
WITH CTE_CUSTOMER_INFERRED_TOWN AS (
    SELECT 
        s.CustomerID,
        s.TownID,
        COUNT(DISTINCT s.deliverydate_) AS N_ORDERS,
        ROW_NUMBER() OVER(PARTITION BY s.CustomerID ORDER BY COUNT(DISTINCT s.deliverydate_) DESC) AS RN
    FROM #customer_rfm c
    LEFT JOIN #BusinessAddressbook b ON c.CustomerID = b.CustomerID
    INNER JOIN #fact_sales_temp s ON c.CustomerID = s.CustomerID
    WHERE b.TownID IS NULL 
      AND s.TownID IS NOT NULL
    GROUP BY s.CustomerID, s.TownID
)
,CTE_TOP_TOWN AS (
    SELECT 
        ci.CustomerID,
        d.CityID, 
        d.CityName, 
        ci.TownID, 
        d.TownName, 
        d.StateID, 
        d.StateName
    FROM CTE_CUSTOMER_INFERRED_TOWN ci
    INNER JOIN #dim_location d ON ci.TownID = d.TownID
    WHERE ci.RN = 1
)
UPDATE b
SET 
    b.CityID = t.CityID,
    b.CityName = t.CityName,
    b.TownID = t.TownID,
    b.TownName = t.TownName,
    b.StateID = t.StateID,
    b.StateName = t.StateName,
	b.MOD_LOCATION = 1
FROM #BusinessAddressbook b
INNER JOIN CTE_TOP_TOWN t ON b.CustomerID = t.CustomerID
WHERE b.TownID IS NULL;

-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
IF OBJECT_ID('tempdb..#dim_customer') IS NOT NULL DROP TABLE #dim_customer;  
SELECT
	b.CustomerID, c.Recency, c.Frequency,	c.Monetary,
	BusinessID,	ContactName,	ContactPhone,	
	StateID,	StateName,	CityID,	CityName,	TownID,	TownName,	
	Latitude,	Longitude,	
	CustomerTypeID,	CustomerModeName,	
	CustomerType,	CustomerStatus
INTO #dim_customer
FROM #customer_rfm c
FULL JOIN #BusinessAddressbook b ON b.CustomerID = c.CustomerID

----------------------------------------------------------------
-- 3. TRANSACTION FACT TABLE
----------------------------------------------------------------
--'Week', 'CustomerID', 'SKUID', 'OrderQty', 'OrderValue', 'TransactionDate' 
IF OBJECT_ID('tempdb..#fact_transaction') IS NOT NULL DROP TABLE #fact_transaction;
SELECT TOP 100  --ItemID as SKUID, SKUCODE	
	DATEPART(ISO_WEEK, deliverydate) AS Week,
    CustomerID,
	ItemID as SKUID, 
	SKUCODE,
	COUNT(DISTINCT OrderID) AS OrderCount,
	COUNT(DISTINCT deliverydate_) AS OrderFreqDays,
	SUM(Quantity) AS OrderQty,	
	SUM(Price) AS OrderValue,
	MAX(deliverydate_) lastDeliveryDate
INTO #fact_transaction
FROM #fact_sales_temp
GROUP BY DATEPART(ISO_WEEK, deliverydate), CustomerID, ItemID, SKUCODE
 
-------------------------------------------------------------------------------
 
 -- Clean up intermediate temp tables
--DROP TABLE #latest_records;
--DROP TABLE #BusinessAddressbook_temp;
--DROP TABLE #dim_location;

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--SELECT TOP 10 * FROM #BusinessAddressbook
--SELECT TOP 10 * FROM #fact_transaction
--SELECT TOP 10 * FROM #dim_sku
--SELECT TOP 10 * FROM #customer_rfm

SELECT COUNT(*) FROM #dim_sku
SELECT COUNT(*) FROM #dim_customer 
SELECT COUNT(*) FROM #fact_transaction

--select top 10 * from segmentmaster where SegmentName = 'Food';
