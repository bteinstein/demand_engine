USE VconnectMasterDWR;

--- 00: StockPoints
drop table if exists #sps
select Stock_Point_ID as StockPointID, Stock_point_Name  StockPointName 
into #sps 
from Stock_Point_Master 
where isnull(status, 0) = 1 and Fulfilement_Center_ID = 76;
 

---- 01: ShipmentIDs
DROP TABLE IF EXISTS #ShipmentIDs; 
CREATE TABLE #ShipmentIDs (
    ShipmentID VARCHAR(20) PRIMARY KEY
);
INSERT INTO #ShipmentIDs (ShipmentID) VALUES
    ('CAU_D9-3401625'),
	('CAU_D1-3392541'),
	('NET_D1-3403314'),
	('CAU_D1-3404847'),
	('REA_D1-3404629'), 
	('OMN_D3-3403842') 
	;

	
---- 02: ShipmentDetails
DROP TABLE IF EXISTS #ShipmentDetails;  
SELECT
    PackageListID as ShipmentID,
    OrderID as ShipmentOrderID,
    CreatedDate as ShipmentCreatedDate,
    DriverAssignedDate,
    FleetType
INTO #ShipmentDetails
FROM OrderPackageList opl WITH (NOLOCK)
WHERE opl.PackageListID IN (SELECT ShipmentID FROM #ShipmentIDs)



---- 03: ShipmentOrderDetails 
DROP TABLE IF EXISTS #ShipmentOrderDetails; 
SELECT
	od.StockPointID,
	sd.ShipmentID,
    od.OrderID as ShipmentOrderID,
    od.Price,
    od.Quantity,
    od.ItemID
INTO #ShipmentOrderDetails
FROM vconnectmasterpos..orderdetails od WITH (NOLOCK)
INNER JOIN #ShipmentDetails sd ON od.OrderID = sd.ShipmentOrderID 

--SELECT SUM(Price) TotPrice, SUM(Quantity) TotQty FROM #ShipmentOrderDetails;
 
---- 04: DriverResale  
DROP TABLE IF EXISTS #DriverResale;
select 
	DISTINCT 
	d.ShipmentID,
	d.SrOrderID as ShipmentOrderID,
	o.customer_id as CustomerID,
	d.OrderID, 
	d.CreatedDate,
	d.SKUID ITEMID, 
	d.hubid StockPointID, 
	max(d.unitprice) soldprice, 
	max(d.Quantity) Quantity, 
	max(d.unitprice)  * max(d.Quantity)  as Price,    
	'Driver Resell' Mode
	--o.CustomerLat,	o.CustomerLong
INTO #DriverResale
FROM DriverResellOrderDetail d WITH (NOLOCK) 
LEFT JOIN DriverResellOrdermaster o ON o.ContentID = d.OrderID
INNER JOIN #ShipmentOrderDetails so ON so.ShipmentOrderID = d.SrOrderID AND so.ShipmentID = d.ShipmentID
WHERE d.STATUS = 1
group by d.ShipmentID, d.orderid, d.SrOrderID,  d.SKUId , d.hubid,  d.CreatedDate, d.ContentId, o.customer_id  



--SELECT 1539780.00/ 1776780.00 AS CRV, 162/178.0 CRQ


	---- 04: DeliveryOrderDetails  
	DROP TABLE IF EXISTS #DeliveryOrderDetails;
	SELECT  
		t.ShipmentID, 
		t.orderid,
		t.MasterOrderID,
		t.CustomerID, 
		t.orderstatusID,
		t.OrderStatus,
		t.deliveredDate,
		t.Mode1 as Mode,
		T.ItemID,
		t.Quantity,
		t.Price  
	INTO #DeliveryOrderDetails 
	FROM tblorderSales t
	INNER JOIN #ShipmentIDs sd ON t.ShipmentID = sd.ShipmentID
	AND OrderStatusID NOT IN (215, 216) 


;WITH ShipmentDetails AS (
	SELECT DISTINCT ShipmentID,ShipmentCreatedDate,DriverAssignedDate,FleetType, COUNT(ShipmentOrderID) nShipmentOrderID FROM #ShipmentDetails GROUP BY ShipmentID,ShipmentCreatedDate,DriverAssignedDate,FleetType
)
, ShipmentOrderDetails AS (
	SELECT StockPointID, ShipmentID, COALESCE(SUM(Price),0) ShipValue, COALESCE(SUM(Quantity),0) ShipQty FROM #ShipmentOrderDetails GROUP BY StockPointID,ShipmentID
)
, DriverResale AS (
	SELECT StockPointID, ShipmentID, 
		COALESCE(SUM(Price),0) PushPrice, COALESCE(SUM(Quantity),0) PushQty, COALESCE(COUNT(DISTINCT CustomerID),0) AS PushCustomers, COALESCE(COUNT(DISTINCT ItemID),0) PushSKUs 
	FROM #DriverResale GROUP BY  StockPointID, ShipmentID
)
, AllSales AS (
	SELECT ShipmentID, 
		COALESCE(SUM(Price),0) PullPrice, COALESCE(SUM(Quantity),0) PullQty, COALESCE(COUNT(DISTINCT CustomerID),0) AS PullCustomers, COALESCE(COUNT(DISTINCT ItemID),0) PullSKUs 
	FROM #DeliveryOrderDetails WHERE Mode = 'Pull' GROUP BY  ShipmentID 
)
SELECT 
	so.StockPointID,     
	sp.StockPointName, 
    sd.ShipmentID, 
	sd.FleetType,
	sd.ShipmentCreatedDate, 
	sd.DriverAssignedDate, 
	ShipValue, ShipQty, 
	ISNULL(PushPrice,0) PushPrice, 
	ISNULL(PushQty,0) PushQty, 
	ISNULL(PushCustomers,0) PushCustomers, 
	ISNULL(PushSKUs,0) PushSKUs,
	ISNULL(PullQty,0) PullQty, 
	ISNULL(PullPrice,0) PullPrice, 
	ISNULL(PullCustomers,0) PullCustomers, 
	ISNULL(PullSKUs,0) PullSKUs,
	COALESCE(PushQty,0) + COALESCE(PullSKUs,0) AS TotSoldQty,
    COALESCE(PushPrice,0) + COALESCE(PullPrice,0) AS TotSoldPrice,
	ROUND((COALESCE(PushQty,0) + COALESCE(PullSKUs,0)) / CAST(COALESCE(ShipQty,0) AS FLOAT), 2) AS ConvRateVol,
	ROUND((COALESCE(PushPrice,0) + COALESCE(PullPrice,0)) / CAST(ShipValue AS FLOAT), 2) AS ConvRateVal
FROM ShipmentDetails sd
INNER JOIN ShipmentOrderDetails so  ON sd.ShipmentID = so.ShipmentID 
LEFT JOIN DriverResale dr ON dr.ShipmentID = so.ShipmentID AND dr.StockPointID = so.StockPointID
LEFT JOIN AllSales al ON sd.ShipmentID = al.ShipmentID
LEFT JOIN #sps sp ON so.StockPointID = sp.StockPointID
ORDER BY ShipmentCreatedDate






--SELECT 
--    so.StockPointID,     sp.StockPointName, 
--    sd.ShipmentID, sd.FleetType,
--	sd.ShipmentCreatedDate, sd.DriverAssignedDate,
--    SUM(so.Quantity) AS ShipQty,
--    SUM(so.Price) AS ShipValue,
--    COUNT(DISTINCT so.ItemID) AS ShipSKUs,
--    SUM(rs.Price) AS PushPrice,
--    SUM(rs.Quantity) AS PushQty,
--    COUNT(DISTINCT rs.ItemID) AS PushSKUs,
--    COUNT(DISTINCT rs.CustomerID) AS PushCustomers,
--    COALESCE(SUM(do.Price), 0) AS PullPrice,
--    COALESCE(SUM(do.Quantity),0) AS PullQty,
--    COUNT(DISTINCT do.ItemID) AS PullSKUs,
--    COUNT(DISTINCT do.CustomerID) AS PullCustomers,
--    COALESCE(SUM(rs.Quantity), 0) + COALESCE(SUM(do.Quantity), 0) AS TotSoldQty,
--    COALESCE(SUM(rs.Price), 0) + COALESCE(SUM(do.Price), 0) AS TotSoldPrice,
--    ROUND(
--        CAST(COALESCE(SUM(rs.Quantity), 0) + COALESCE(SUM(do.Quantity), 0) AS FLOAT) / 
--        NULLIF(SUM(so.Quantity), 0), 
--        2 ) AS ConvRateVol,
--    ROUND(
--        CAST(COALESCE(SUM(rs.Price), 0) + COALESCE(SUM(do.Price), 0) AS FLOAT) / 
--        NULLIF(SUM(so.Price), 0), 
--        2) AS ConvRateVal
--FROM #ShipmentDetails sd
--LEFT JOIN #ShipmentOrderDetails so ON sd.ShipmentID = so.ShipmentID AND sd.ShipmentOrderID = so.ShipmentOrderID
--LEFT JOIN #DriverResale rs ON sd.ShipmentID = rs.ShipmentID AND sd.ShipmentOrderID = rs.ShipmentOrderID AND so.ItemID = rs.ItemID
--LEFT JOIN #DeliveryOrderDetails do ON sd.ShipmentID = do.ShipmentID AND sd.ShipmentOrderID = do.OrderID AND so.ItemID = do.ItemID AND do.OrderStatusID = 214  -- Uncomment if 214 is correct
--LEFT JOIN #sps sp ON so.StockPointID = sp.StockPointID
--GROUP BY so.StockPointID, sp.StockPointName, sd.ShipmentID, sd.FleetType, ShipmentCreatedDate, DriverAssignedDate
--ORDER BY sd.ShipmentCreatedDate ;


--SELECT 
--    so.StockPointID,     sp.StockPointName, 
--    sd.ShipmentID, sd.FleetType,
--	sd.ShipmentCreatedDate, sd.DriverAssignedDate,
--    SUM(so.Quantity) AS ShipQty,
--    SUM(so.Price) AS ShipValue,
--    COUNT(DISTINCT so.ItemID) AS ShipSKUs,
--    SUM(rs.Price) AS PushPrice,
--    SUM(rs.Quantity) AS PushQty,
--    COUNT(DISTINCT rs.ItemID) AS PushSKUs,
--    COUNT(DISTINCT rs.CustomerID) AS PushCustomers,
--    COALESCE(SUM(do.Price), 0) AS PullPrice,
--    COALESCE(SUM(do.Quantity),0) AS PullQty,
--    COUNT(DISTINCT do.ItemID) AS PullSKUs,
--    COUNT(DISTINCT do.CustomerID) AS PullCustomers,
--    COALESCE(SUM(rs.Quantity), 0) + COALESCE(SUM(do.Quantity), 0) AS TotSoldQty,
--    COALESCE(SUM(rs.Price), 0) + COALESCE(SUM(do.Price), 0) AS TotSoldPrice,
--    ROUND(
--        CAST(COALESCE(SUM(rs.Quantity), 0) + COALESCE(SUM(do.Quantity), 0) AS FLOAT) / 
--        NULLIF(SUM(so.Quantity), 0), 
--        2 ) AS ConvRateVol,
--    ROUND(
--        CAST(COALESCE(SUM(rs.Price), 0) + COALESCE(SUM(do.Price), 0) AS FLOAT) / 
--        NULLIF(SUM(so.Price), 0), 
--        2) AS ConvRateVal
--FROM #ShipmentDetails sd
--LEFT JOIN #ShipmentOrderDetails so ON sd.ShipmentID = so.ShipmentID AND sd.ShipmentOrderID = so.ShipmentOrderID
--LEFT JOIN #DriverResale rs ON sd.ShipmentID = rs.ShipmentID AND sd.ShipmentOrderID = rs.ShipmentOrderID AND so.ItemID = rs.ItemID
--LEFT JOIN #DeliveryOrderDetails do ON sd.ShipmentID = do.ShipmentID AND sd.ShipmentOrderID = do.OrderID AND so.ItemID = do.ItemID AND do.OrderStatusID = 214  -- Uncomment if 214 is correct
--LEFT JOIN #sps sp ON so.StockPointID = sp.StockPointID
--GROUP BY so.StockPointID, sp.StockPointName, sd.ShipmentID, sd.FleetType, ShipmentCreatedDate, DriverAssignedDate
--ORDER BY sd.ShipmentCreatedDate ;




--SELECT TOP 5 * FROM #ShipmentDetails;
--SELECT TOP 5 * FROM #ShipmentOrderDetails;
--SELECT TOP 5 * FROM #DriverResale;
--SELECT TOP 5 * FROM #DeliveryOrderDetails ORDER BY ShipmentID, Mode;

