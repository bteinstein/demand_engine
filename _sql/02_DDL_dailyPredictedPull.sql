USE VCONNECTMASTERDWR;

--
CREATE TABLE dbo.dailyPredictedPull (
    ID                        INT IDENTITY(1,1) PRIMARY KEY,
    StockPointID              INT,
    StockPointName            NVARCHAR(60),
    TripID                    INT,
    ClusterLGAs               NVARCHAR(1000),
    ClusterLCDAs              NVARCHAR(1000),
    TotalCustonerCount        INT,
    TripTotalQuantity         INT,
    TripAvgCustomerScore      FLOAT,
    CustomerID                INT,
    ContactName               NVARCHAR(100),
    CustomerModeName          NVARCHAR(50),
    ContactPhone              NVARCHAR(20),
    FullAddress               NVARCHAR(500),
    Latitude                  FLOAT,
    Longitude                 FLOAT,
    LGA                       NVARCHAR(50),
    LCDA                      NVARCHAR(50),
    CustomerScore             FLOAT,
    kycCaptureStatus          NVARCHAR(3),
    SKUID                     INT,
    ProductName               NVARCHAR(100),
    SKUDaysSinceLastBuy       FLOAT,
    CustomerDaysSinceLastBuy  INT,
    InventoryCheck            NVARCHAR(20),
    ProductTag                NVARCHAR(20),
    RecommendationType        NVARCHAR(15),
    EstimatedQuantity         INT,
    isTripSelected            NVARCHAR(3),
    ModifiedDate              DATE
);

-- For upsert match condition
CREATE UNIQUE INDEX IX_dailyPredictedPull_UniqueUpsert
ON dbo.dailyPredictedPull (StockPointID, CustomerID, SKUID, ModifiedDate);

-- Likely filter columns (optional, based on your use case)
CREATE INDEX IX_dailyPredictedPull_TripID
ON dbo.dailyPredictedPull (TripID);

CREATE INDEX IX_dailyPredictedPull_StockPointID
ON dbo.dailyPredictedPull (StockPointID);

CREATE INDEX IX_dailyPredictedPull_CustomerID
ON dbo.dailyPredictedPull (CustomerID);



--SELECT TOP 100 * FROM dailyPredictedPull;
--DROP TABLE IF EXISTS dailyPredictedPullClusterSummary;
--TRUNCATE TABLE   dailyPredictedPullClusterSummary;
CREATE TABLE dbo.dailyPredictedPullClusterSummary (
    ID                        INT IDENTITY(1,1) PRIMARY KEY,
    StockPointID              INT,
    StockPointName            NVARCHAR(60),
    TripID                    INT,
    ClusterLGAs               NVARCHAR(1000),
    ClusterLCDAs              NVARCHAR(1000),
    TotalCustonerCount        INT,
    TripTotalQuantity         INT,
    TripAvgCustomerScore      FLOAT,
    ModifiedDate              DATE
);

 -- For upsert match condition
CREATE UNIQUE INDEX IX_dailyPredictedPullClusterSummary_UniqueUpsert
ON dbo.dailyPredictedPullClusterSummary (StockPointID, TripID,  ModifiedDate);

-- Likely filter columns (optional, based on your use case)
CREATE INDEX IX_dailyPredictedPullClusterSummary_TripID
ON dbo.dailyPredictedPullClusterSummary (TripID);

CREATE INDEX IX_dailyPredictedPullClusterSummary_StockPointID
ON dbo.dailyPredictedPullClusterSummary (StockPointID);

ALTER TABLE dbo.dailyPredictedPullClusterSummary
ALTER COLUMN ClusterLGAs NVARCHAR(1000);

ALTER TABLE dbo.dailyPredictedPullClusterSummary
ALTER COLUMN ClusterLCDAs NVARCHAR(1000);

SELECT TOP 100 * FROM dailyPredictedPullClusterSummary;