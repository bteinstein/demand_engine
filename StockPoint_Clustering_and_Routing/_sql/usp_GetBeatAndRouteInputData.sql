/*  EXEC usp_GetBeatAndRouteInputData;  */  
CREATE OR ALTER PROCEDURE usp_GetBeatAndRouteInputData  
AS 
BEGIN     
    SET NOCOUNT ON;              

    -------------------------------------------------------------------------
    -- SP LOCATION MAP
    -------------------------------------------------------------------------
    DROP TABLE IF EXISTS #spLocationMapping;

    SELECT DISTINCT
          fm.Fulfilement_Center_ID AS Stock_Point_ID,
          sp.Stock_point_Name AS Stock_Point_Name,
          sm.StateName AS State_Name,
          cm.StateID AS State_ID,
          lgm.Region,
          cm.CityName AS LGA_Name,
          tm.CityID AS LGA_ID,
          tm.TownName AS LCDA_Name,
          fm.Location_ID AS LCDA_ID
    INTO #spLocationMapping
    FROM VCONNECTMASTERDWR..FC_Location_Mapping fm WITH (NOLOCK)
    INNER JOIN VCONNECTMASTERDWR..Townmaster tm WITH (NOLOCK) 
        ON fm.location_id = tm.Contentid
    INNER JOIN VCONNECTMASTERDWR..citymaster cm WITH (NOLOCK) 
        ON tm.CityID = cm.Contentid
    INNER JOIN VCONNECTMASTERDWR..statemaster sm WITH (NOLOCK) 
        ON cm.stateid = sm.Contentid
    LEFT JOIN VCONNECTMASTERDWR..LGA_Region_Master lgm WITH (NOLOCK) 
        ON lgm.LGA_ID = tm.CityID
    INNER JOIN VCONNECTMASTERDWR..Stock_Point_Master sp WITH (NOLOCK) 
        ON fm.Fulfilement_Center_ID = sp.Stock_Point_ID
        AND sp.Fulfilement_Center_ID = 76
    WHERE fm.[status] = 1 -- Active
      AND fm.Location_Type = 3 -- LCDA
      AND lgm.CountryID = 1
      AND (Is_Fulfilement_Center = 1 OR Is_Mfc = 1 OR Is_Mfc = 0)
      AND sp.Stock_point_Name NOT LIKE '%Test%'
      AND ISNULL(sp.Is_Mfc,0) = 1
      AND ISNULL(sp.Status,0) = 1;

    -------------------------------------------------------------------------
    -- SP Dim
    -------------------------------------------------------------------------
    DROP TABLE IF EXISTS #spDim;

    SELECT
          sp.Stock_Point_ID,
          sp.Stock_point_Name,
          bm.Lattitude AS Latitude,
          bm.Longitude
    INTO #spDim
    FROM Stock_Point_Master sp WITH (NOLOCK)
    INNER JOIN BusinessMaster bm WITH (NOLOCK) 
        ON sp.Stock_Point_ID = bm.Contentid
    WHERE sp.Fulfilement_Center_ID = 76
      AND sp.Status = 1
      AND sp.Warehouse_Type IN (1, 3)
      AND sp.Stock_Point_Name NOT LIKE '%TEST%'
      AND sp.Is_Mfc = 1;

    -------------------------------------------------------------------------
    -- Active Customers (DU: 21,487)
    -------------------------------------------------------------------------
    DROP TABLE IF EXISTS #ActiveCustomers;

    DECLARE @ActiveStartDate DATE = '2025-01-01'; -- DATEADD(month, -6, GETDATE());

    SELECT
          Stock_Point_ID,
          CustomerID AS Customer_ID,
          MIN(deliverydate) AS first_delv_date,
          MAX(deliverydate) AS last_delv_date
    INTO #ActiveCustomers
    FROM (
        SELECT
              BusinessID AS Stock_Point_ID,
              CustomerID,
              deliverydate
        FROM VCONNECTMASTERDWR..tblmanudashsales WITH (NOLOCK)
        WHERE Central_BusinessID = 76
          AND deliverydate >= @ActiveStartDate
    ) A
    GROUP BY Stock_Point_ID, CustomerID;

    -------------------------------------------------------------------------
    -- Customer Dim (31,969)
    -------------------------------------------------------------------------
    DROP TABLE IF EXISTS #CustomerDim;

    DECLARE @6MonthsAgo DATE = DATEADD(month, -6, GETDATE());

    SELECT
          c.CustomerID AS Customer_ID,
          c.BusinessID AS Business_ID,
          CAST(c.CustomerCreatedDate AS DATE) AS Created_Date,
          c.ContactName AS Contact_Name,
          c.ContactPhone AS Contact_Phone,
          c.Statename AS State_Name,
          c.TownName AS Town_Name,
          c.CityName AS City_Name,
          c.Latitude,
          c.Longitude,
          c.CustomerStatus AS Customer_Status,
          c.Status,
          c.FirstName AS First_Name,
          ISNULL(c.IsLocationCaptured, 0) AS Is_Location_Captured,
          ISNULL(c.IsLocationSubmitted, 0) AS Is_Location_Submitted,
          c.LocationSubmittedDate AS Location_Submitted_Date,
          ISNULL(c.IsLocationVerified, 0) AS Is_Location_Verified,
          c.LocationVerifiedDate AS Location_Verified_Date,
          c.Location,
          c.FullAddress AS Full_Address,
          c.KYC_Capture_Status,
          c.AgentID AS Agent_ID,
          c.AgentName AS Agent_Name
    INTO #CustomerDim
    FROM VConnectMasterDWR..CustomerKYCdump AS c WITH (NOLOCK)
    WHERE c.BusinessID = 76
      AND (
            -- Condition 1: The customer is an active customer
            EXISTS (
                SELECT 1
                FROM #ActiveCustomers ac
                WHERE ac.Customer_ID = c.CustomerID
            )
            -- OR Condition 2: The customer was created within the last 6 months
            OR c.CustomerCreatedDate >= @6MonthsAgo
          );
    
     

    --------------------- AGENT-CUSTOMER DIM TABLE --------------       
    DROP TABLE IF EXISTS #AgentCustomer                       
    CREATE TABLE #AgentCustomer                      
    (                      
     Agent_ID int NULL                      
    ,Agent_Name nvarchar(300) NULL                      
    ,Role_ID int NULL                      
    ,Role_Name nvarchar(300) NULL                  
    ,Customer_ID INT NOT NULL                      
    ---,ContactName nvarchar(300) NULL                  
    )                
      
    CREATE NONCLUSTERED INDEX #IX_AGENT_CUSTOMER ON #AgentCustomer (CUSTOMER_ID,AGENT_ID);             

    INSERT INTO #AgentCustomer (AGENT_ID, AGENT_NAME,ROLE_ID,ROLE_NAME,CUSTOMER_ID  )        
     SELECT DISTINCT B.USERID, B.AGENTNAME, B.ROLEID, R.ROLENAME, A.CUSTOMER_ID          
     FROM #CustomerDim A      
     LEFT JOIN BUSINESSUSERROLE B WITH (NOLOCK) ON A.AGENT_ID = B.USERID AND B.STATUS = 1 AND B.BUSINESSID = 76         
     LEFT JOIN ROLEMASTER R ON B.ROLEID = R.CONTENTID AND R.STATUS = 1  
     WHERE R.contentid in (18235);

     /*
     
     SELECT TOP 10 * FROM #AgentCustomer
     SELECT DISTINCT ROLE_ID, ROLE_NAME FROM #AgentCustomer
     SELECT COUNT(DISTINCT AGENT_ID) FROM #AgentCustomer --- 774
        
    SELECT COUNT(DISTINCT AGENT_ID) FROM #AgentCustomer 
    WHERE CUSTOMER_ID IN (SELECT CUSTOMER_ID FROM #ActiveCustomers WHERE STOCK_POINT_ID = 1647113)
     
     SELECT * FROM ROLEMASTER WHERE CONTENTID IN (18235,16769)
     WHERE ROLENAME LIKE '%OAM%'

     SELECT TOP 10 
              BusinessID AS Stock_Point_ID,
              CustomerID,
              deliverydate, *
        FROM VCONNECTMASTERDWR..tblmanudashsales WITH (NOLOCK)
        WHERE Central_BusinessID = 76

    ;WITH CTE_ACTIVE_CUSTOMERS AS (
    SELECT DISTINCT CustomerID AS Customer_ID
    FROM VCONNECTMASTERDWR..tblmanudashsales WITH (NOLOCK)
    WHERE Central_BusinessID = 76 AND deliverydate >= '2025-01-01' AND BUSINESSID = 1647113
    ),
    CTE_CUSTOMER_DIM AS (
        SELECT
            c.CustomerID AS Customer_ID,
            c.AgentID AS Agent_ID,
            c.AgentName AS Agent_Name
        FROM VConnectMasterDWR..CustomerKYCdump AS c WITH (NOLOCK)
        WHERE
            c.BusinessID = 76
            AND EXISTS (
                SELECT 1 FROM CTE_ACTIVE_CUSTOMERS ac WHERE ac.Customer_ID = c.CustomerID
            )
    )
    SELECT
        COUNT(DISTINCT B.USERID) AS N_OAM
    FROM
        CTE_CUSTOMER_DIM A
    INNER JOIN
        BUSINESSUSERROLE B WITH (NOLOCK) ON A.Agent_ID = B.USERID
        AND B.STATUS = 1
        AND B.BUSINESSID = 76
    INNER JOIN
        ROLEMASTER R ON B.ROLEID = R.CONTENTID
        AND R.STATUS = 1
    WHERE
        R.contentid = 18235;

       EXEC SP_HELPTEXT sp_GetCustomerSKUSalesRecommendationALGO
       EXEC SP_HELPTEXT usp_GetCustomerKYCInfoDetailsV2
     */

    -------------------------------------------------------------------------
    -- Return result sets
    -------------------------------------------------------------------------
    -- SELECT 'spLocationMapping' AS TableName, 1 AS ResultOrder;
    SELECT * FROM #spLocationMapping;

    -- SELECT 'spDim' AS TableName, 2 AS ResultOrder;
    SELECT * FROM #spDim;

    -- SELECT 'ActiveCustomers' AS TableName, 3 AS ResultOrder;
    SELECT * FROM #ActiveCustomers;

    -- SELECT 'CustomerDim' AS TableName, 4 AS ResultOrder;
    SELECT * FROM #CustomerDim;

    SELECT * FROM #AgentCustomer;

END;
