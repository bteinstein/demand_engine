--USE VconnectMasterDWR;
--EXEC SP_HELPTEXT usp_GetCustomerKYCInfoDetails;


CREATE OR ALTER PROCEDURE [dbo].[usp_GetCustomerKYCInfoDetailsV2]    
AS  
BEGIN  
    SET NOCOUNT ON;  
    SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;  
  
 -----------------------------------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#agency') IS NOT NULL DROP TABLE #agency;   
 ;WITH CTE as (  
  SELECT UserID, AgentName, RoleName,  
  ROW_NUMBER() OVER(PARTITION BY Userid, AgentName ORDER BY b.Contentid desc) as ranker  
  FROM BusinessUserRole b WITH (NOLOCK)  
  LEFT JOIN ROleMaster r on b.RoleID  = r.Contentid  
  WHERE RoleID in (16769, 26106)   
 )  
 SELECT * INTO #agency  
 FROM CTE  
 WHERE ranker = 1  
  
 -----------------------------------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#DimCustRefID') IS NOT NULL DROP TABLE #DimCustRefID;  
 SELECT DISTINCT  
   CustomerID, CustomerRef  
  INTO #DimCustRefID  
  FROM VConnectMasterDWR.dbo.OmniPayCustomerWallets WITH (NOLOCK)  
  WHERE BusinessId = 76;  
  
 IF OBJECT_ID('tempdb..#dimHasPOS') IS NOT NULL DROP TABLE #dimHasPOS;  
 SELECT  
  DISTINCT CustomerID, 1 AS hasPOS  
 INTO #dimHasPOS  
 FROM VconnectMasterPOS..OmniPayOrderReport P WITH (NOLOCK)    
 INNER JOIN #DimCustRefID C ON C.CustomerRef = P.CustomerRef  
 WHERE P.event IN ('pos.delivered', 'posdeposit.successful')  
  
 -----------------------------------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#dimHasBNPL') IS NOT NULL DROP TABLE #dimHasBNPL;    
 ;WITH lastLimitDate AS (  
  SELECT  
   CustomerID,  
   P.CustomerRef,  
   MAX(CreatedAt) as max_date  
  FROM VconnectMasterPOS..OmniPayOrderReport P WITH (NOLOCK)  
  INNER JOIN #DimCustRefID C ON C.CustomerRef = P.CustomerRef  
  WHERE P.event IN ('paylaterlimit.downgraded',  
        'paylaterlimit.updated', 'paylaterlimit.upgraded')  
  GROUP BY CustomerID, P.CustomerRef  
 )  
 SELECT  
  CustomerID,  
  CASE WHEN Amount > 0 THEN 1 ELSE 0 END AS hasBNPL  
 INTO #dimHasBNPL  
 FROM VconnectMasterPOS..OmniPayOrderReport P WITH (NOLOCK)  
 INNER JOIN lastLimitDate C ON C.CustomerRef = P.CustomerRef AND CreatedAt = max_date;  
   
 ------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#dimHasVAS') IS NOT NULL DROP TABLE #dimHasVAS;   
 SELECT  
  CustomerID, 1 as HasVAS  
 INTO #dimHasVAS   
 FROM (  
  SELECT  
   CustomerID,  
   LOWER(CustomerModeName) AS VAS,  
   ROW_NUMBER() OVER (PARTITION BY CustomerID ORDER BY ModifiedDate DESC) AS RN  
  FROM OmniDR..AddressBook WITH (NOLOCK)  
  WHERE  
   CustomerModeName LIKE 'omnishelf champion'  
   AND ISNULL(status, 0) IN (0, 1, 2, 7)  
   AND BusinessID = 76  
 ) V  
 WHERE RN = 1;  
  
  
 ------------------------------------------------------------  
 -- ALL DELIVERED ORDER FROM 2024-01-01  
 ------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#dimDevl') IS NOT NULL DROP TABLE #dimDevl;   
 SELECT  
  CustomerID,  
  MAX(DeliveryDate) AS lastDelvDate  
 INTO #dimDevl  
 FROM tblManuDashSales    
 WHERE Central_Businessid =76  
   AND DeliveryDate >= '2023-01-01'  -- Filter for orders delivered from January 2024 onwards  
 GROUP BY CustomerID;  
 ------------------------------------------------------------------------------------------------------------------------------------------------------------------  
  
 -- First drop the table if it exists  
 IF OBJECT_ID('tempdb..#CustomerListVerficationStatus') IS NOT NULL DROP TABLE #CustomerListVerficationStatus;  
  
 -- Create the table with your exact data types and length specifications  
 CREATE TABLE #CustomerListVerficationStatus (  
  CustomerID int NULL,  
  ContactName varchar(512) NULL,  
  BusinessName varchar(512) NULL,  
  CustomerModeName varchar(128) NULL,  
  CustomerRef nvarchar(400) NULL,  
  ContactPhone varchar(48) NULL,  
  CustomerType varchar(96) NULL,  
  CustomerCreatedDate varchar(50) NULL,  
  AgentID int NULL,  
  AgentName varchar(128) NULL,  
  AgentIDs varchar(2048) NULL,  
  AgentNames varchar(2048) NULL,  
  Location varchar(1024) NULL,  
  Address varchar(512) NULL,  
  FullAddress varchar(1024) NULL,  
  StateName varchar(48) NULL,  
  CityName varchar(48) NULL,  
  TownName varchar(96) NULL,  
  Latitude varchar(128) NULL,  
  Longitude varchar(128) NULL,  
  status int NULL,  
  FirstName varchar(512) NULL,  
  LastName varchar(512) NULL,  
  modifiedBy int NULL,  
  ExternalAgentName varchar(128) NULL,  
  Exteragentid int NULL,  
  ExteragentRole varchar(96) NULL,  
  DistanceVarianceInMeter float NULL,  
  IsLocationSubmitted varchar(3) NOT NULL,  
  LocationSubmittedDate datetime NULL,  
  IsLocationCaptured varchar(3) NULL,  
  IsLocationVerified varchar(3) NULL,  
  LocationVerifiedDate datetime NULL,  
  RecaptureCount int NULL,  
  CustomerStatus varchar(13) NULL,  
  RejectReason varchar(512) NULL,  
  RejectionDate datetime NULL  
 );  
  
 -- Create indexes before data insertion  
 CREATE NONCLUSTERED INDEX IX_CustomerListVS_CustomerID ON #CustomerListVerficationStatus (CustomerID);  
 CREATE NONCLUSTERED INDEX IX_CustomerListVS_Status ON #CustomerListVerficationStatus (status, CustomerStatus);  
 CREATE NONCLUSTERED INDEX IX_CustomerListVS_Agents ON #CustomerListVerficationStatus (AgentID, modifiedBy, Exteragentid);  
 CREATE NONCLUSTERED INDEX IX_CustomerListVS_Location ON #CustomerListVerficationStatus (IsLocationVerified, IsLocationCaptured, IsLocationSubmitted);  
 CREATE NONCLUSTERED INDEX IX_CustomerListVS_CustomerRef ON #CustomerListVerficationStatus (CustomerRef);  
  
 -- Now insert the data using your existing query  
 INSERT INTO #CustomerListVerficationStatus  
 SELECT   
  b.CustomerID,    
  b.ContactName,   
  b.BusinessName,   
  b.CustomerModeName,   
  b.CustomerRef,  
  b.ContactPhone,   
  b.CustomerType,   
  b.CustomerCreatedDate,  
  b.AgentID,   
  b.AgentName,   
  b.AgentIDs,   
  b.AgentNames,  
  b.[Location],   
  COALESCE(NULLIF(b.Address1, ''), b.Address2) AS [Address],  
  b.FullAddress,  
  b.StateName,   
  b.CityName,   
  b.TownName,  
  b.Latitude,   
  b.Longitude,  
  b.status,  
  b.FirstName,   
  b.LastName,  
  b.modifiedBy,  
  b.ExternalAgentName,  
  b.Exteragentid,  
  b.ExteragentRole,  
  b.DistanceVarianceInMeter,  
  CASE   
   WHEN ISNULL(b.IsLocationSubmitted, 0) = 1 THEN 'Yes'   
   ELSE 'No'   
  END AS IsLocationSubmitted,   
  b.LocationSubmittedDate,  
  CASE   
   WHEN ISNULL(b.IsLocationCaptured, 0) = 1 THEN 'Yes'   
   ELSE 'No'   
  END AS IsLocationCaptured,  
  CASE   
   WHEN ISNULL(b.IsLocationVerified, 0) = 1 THEN 'Yes'   
   ELSE 'No'   
  END AS IsLocationVerified,   
  b.LocationVerifiedDate,  
  b.Recapture_Location_Count AS RecaptureCount,  
  CASE   
   WHEN b.status = 0 THEN 'Pending'  
   WHEN b.status = 1 THEN 'Active'  
   WHEN b.status = 2 THEN 'Rejected'  
   WHEN b.status = 7 THEN 'Pending Image'  
  END AS CustomerStatus,  
  b.RejectReason,  
  TRY_CONVERT(DATETIME, LEFT(b.RejectDate, 16) + ':00', 126) AS RejectionDate  
 FROM (  
  SELECT   
   a.*,   
   c.CustomerRef,    
   ag.agentname as ExternalAgentName,  
   ag.Userid as Exteragentid,   
   ag.Rolename as ExteragentRole,  
   v.DistanceVarianceInMeter,  
   ROW_NUMBER() OVER (PARTITION BY a.CustomerID ORDER BY a.ContentID DESC) AS rn  
  FROM OmniDR..AddressBook a WITH (NOLOCK)  
  INNER JOIN #dimDevl dv ON dv.CustomerID = a.CustomerID 
  LEFT JOIN #DimCustRefID c ON c.CustomerID = a.CustomerID   
  LEFT JOIN #agency ag ON a.ModifiedBy = ag.UserID  
  LEFT JOIN OmniDR..kyc_location_variance_check v WITH (NOLOCK) ON a.customerid = v.customerid  
  WHERE   
   a.businessid = 76   
   AND ISNULL(a.Status, 0) IN (0, 1, 2, 7)  
 ) b  
 WHERE b.rn = 1;  
  
  
  
  
 ----------------------- 2. ADDING POC CUSTOMERS TAG ------------------------------  
 -------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#POCcustomerList') IS NOT NULL DROP TABLE #POCcustomerList;  
 SELECT    
  Customer_ID as CustomerID, Customer_Name, CONCAT('POC - ',POC) AS POCbatch, 'Yes' AS POCtag  
 INTO #POCcustomerList  
 FROM VConnectMasterDWR..Customer_Verification_list  WITH (NOLOCK);  
  
  
 -------------------------------------------------------------------------------------------------------------------  
 ----------------------- 3. BVN  ------------------------------   
 -------------------------------------------------------------------------------------------------------------------    
 IF OBJECT_ID('tempdb..#BVNstatu') IS NOT NULL DROP TABLE #BVNstatus;  
 WITH CTE_AgentMapping AS (  
 SELECT DISTINCT    
  a.Userid as customerid, a.[Name] as customer_name, a.Agentid, b.AgentName   
 FROM VConnectMasterDWR..businessaddressbook as a with (nolock)   
 LEFT JOIN VConnectMasterDWR..BusinessUserRole as b with(nolock) ON a.Agentid = b.UserID   
 WHERE b.RoleID=18235 AND a.Businessid=76 AND ISNULL(b.Status, 0) IN (0,1)   
   AND a.Contentid = (SELECT MAX(Contentid) FROM VConnectMasterDWR..BusinessAddressbook AS Sub1 WHERE Sub1.Userid = a.Userid)   
   AND b.Contentid = (SELECT MAX(Contentid) FROM VConnectMasterDWR..BusinessUserRole AS Sub1 WHERE Sub1.Userid = b.Userid)   
 )  
 SELECT    
  b.CustomerID, b.customerRef,   
  1 AS bvnverified,  
  MAX(a.createdAt) AS BVNlatestDate,  
  MAX(d.AgentName) AS OnboardingAgentName,  
  MAX(d.AgentID) AS OnboardingAgentID  
 INTO #BVNstatus  
 FROM #CustomerListVerficationStatus b  
 INNER JOIN VConnectMasterDWR.dbo.OmniPayCustomerWallets c WITH (NOLOCK) ON c.CustomerID = b.CustomerID  
 INNER JOIN VConnectMasterpos..OmnipayorderReport a WITH (NOLOCK) ON a.customerRef = c.customerRef  
 LEFT JOIN CTE_AgentMapping d ON c.CreatedBy = d.Agentid  
 WHERE   
  a.event = 'bvn.verified'  
  AND a.customerRef IS NOT NULL AND c.BusinessId = 76  
 GROUP BY  b.CustomerID, b.customerRef;   
  
  
 --------------------------------------------------------------------------------------------------------------------  
 ---- TABLE BVN: BVNPhonemismatch   
 IF OBJECT_ID('tempdb..#BVNPhonemismatch') IS NOT NULL DROP TABLE #BVNPhonemismatch;  
 WITH CTE_AgentMapping AS (  
  SELECT DISTINCT  
   a.Userid AS customerid, a.[Name] AS customer_name, a.Agentid, b.AgentName  
  FROM VConnectMasterDWR..businessaddressbook AS a WITH (NOLOCK)  
  LEFT JOIN VConnectMasterDWR..BusinessUserRole AS b WITH (NOLOCK) ON a.Agentid = b.UserID  
  WHERE b.RoleID = 18235 AND a.Businessid = 76 AND ISNULL(b.Status, 0) IN (0, 1)  
   AND a.Contentid = ( SELECT MAX(Contentid) FROM VConnectMasterDWR..BusinessAddressbook AS Sub1 WITH (NOLOCK) WHERE Sub1.Userid = a.Userid )  
   AND b.Contentid = ( SELECT MAX(Contentid) FROM VConnectMasterDWR..BusinessUserRole AS Sub2 WITH (NOLOCK) WHERE Sub2.Userid = b.Userid )  
 )  
 SELECT  
  b.CustomerID,  
  b.customerRef,  
  1 AS bvnphonemismatch,  
  MAX(a.createdAt) AS BVNphonemismatchDate,  
  MAX(d.AgentName) AS OnboardingAgentName,  
  MAX(d.AgentID) AS OnboardingAgentID  
 INTO #BVNPhonemismatch  
 FROM #CustomerListVerficationStatus AS b  
 INNER JOIN VConnectMasterDWR.dbo.OmniPayCustomerWallets c WITH (NOLOCK) ON c.CustomerID = b.CustomerID  
 INNER JOIN VConnectMasterpos..OmnipayorderReport AS a WITH (NOLOCK) ON a.customerRef = c.customerRef  
 LEFT JOIN CTE_AgentMapping AS d  ON c.CreatedBy = d.Agentid  
 WHERE  
  a.event = 'bvn.verified.phonemismatch'  
  AND a.customerRef IS NOT NULL AND c.BusinessId = 76  
 GROUP BY b.customerRef, b.CustomerID;  
  
  
 -------------------------------------------------------------------------------------------------------------------  
 ----------------------- 3. IMAGE CAPTURE -------------------------------   
 -------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#ImageUpdate') IS NOT NULL DROP TABLE #ImageUpdate;  
 WITH CTE AS (  
  SELECT    
   img.CustomerID,   
   Count(imagetypeid) ImageTypeIDCount,   
   MAX(COALESCE(img.ModifiedDate, CreatedDate)) AS LastImageUpdatedDate,  
   MAX(COALESCE(img.ModifiedBy, img.AgentID)) AS LastImageAgentID,  
   MAX(CASE WHEN ImageTypeID  = 84 then image Else null END)as ImageUrl84,  
   MAX(CASE WHEN ImageTypeID  = 86 then image Else null END)as ImageUrl86,  
   MAX(CASE WHEN ImageTypeID  = 87 then image Else null END)as ImageUrl87  
  FROM VConnectMasterdwr..BusinessCustomerIMAGE img WITH (NOLOCK)  
  INNER JOIN #CustomerListVerficationStatus c WITH (NOLOCK) ON c.CustomerID = img.CustomerID  
  WHERE ISNULL(img.Status, 0) = 1   
    AND ImageTypeID IN (84, 86, 87)  
    AND ISNULL(img.IMAGE,'') <> ''  
    GROUP  BY img.CustomerID  
 )  
 SELECT *,    
 CASE WHEN ImageTypeIDCount >= 3  AND ISNULL(ImageUrl84,'') <> '' AND ISNULL(ImageUrl86,'') <> '' AND ISNULL(ImageUrl87,'') <> ''  
 THEN 1  ELSE 0   
 END AS ImageCaptured  
 into #ImageUpdate  
 FROM CTE   
  
 -------------------------------------------------------------------------------------------------------------------  
 ------------- FINAL TABLE ---------  
 -------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#FinalTable') IS NOT NULL DROP TABLE #FinalTable;  
 SELECT   
  A.*,   
  -- POC  
  poc.POCbatch, ISNULL(poc.POCtag,'No') AS POCtag,  
  -- BVN  
  CASE WHEN ISNULL(bvn.bvnverified, 0) = 1 THEN 'Yes' ELSE 'No' END AS isBVNverfied,  
  CASE WHEN ISNULL(bvp.bvnphonemismatch, 0) = 1 THEN 'Yes' ELSE 'No' END AS isBVNPhonemismatch  
  ,   
  bvn.BVNlatestDate, bvn.OnboardingAgentName, bvn.OnboardingAgentID,  
  -- IMAGE  
  img.ImageUrl84 as imgURLinsideOutlet, img.ImageUrl86 as imgURLoutletShelf, img.ImageUrl87 as imgURLoutsideOutlet,  
  img.LastImageUpdatedDate, img.LastImageAgentID, img.ImageCaptured  
 INTO #FinalTable  
 FROM #CustomerListVerficationStatus A  
 LEFT JOIN #POCcustomerList poc ON poc.CustomerID = A.CustomerID   
 LEFT JOIN #BVNstatus bvn ON bvn.CustomerID = A.CustomerID   
 LEFT JOIN #ImageUpdate img ON img.CustomerID = A.CustomerID  
 LEFT JOIN #BVNPhonemismatch bvp  ON  bvp.CustomerID = A.CustomerID;    
  
 CREATE NONCLUSTERED INDEX IX_FinalTable_CustomerID ON #FinalTable (CustomerID);  
 -------------------------------------------------------------------------------------------------------------------  
 ------------- DATE HARMONIZATION ---------  
 -------------------------------------------------------------------------------------------------------------------  
 IF OBJECT_ID('tempdb..#FinalCustomerKYCInfo') IS NOT NULL DROP TABLE #FinalCustomerKYCInfo;  
  WITH LatestUpdateDate AS (  
  SELECT   
   CustomerID,  
   MAX(MaxDate) AS LatestUpdateDate  
  FROM (  
   SELECT   
    t.CustomerID,  
    (SELECT MAX(dt)   
     FROM (VALUES   
        (t.CustomerCreatedDate),   
        (t.LocationSubmittedDate),   
        (t.LocationVerifiedDate),   
        (t.RejectionDate),   
        (t.BVNlatestDate),   
        (t.LastImageUpdatedDate)  
        ) AS v(dt)  
    ) AS MaxDate  
   FROM #FinalTable t  
  ) sub  
  GROUP BY CustomerID  
 )  
 SELECT   
  AB.*,   
  lt.LatestUpdateDate,  
  CASE WHEN ISNULL(ab.FirstName,'') <> ''                                      
      AND ISNULL(COALESCE(AB.FullAddress, AB.Location),'') <> ''                                      
      AND ISNULL(AB.IsLocationCaptured,'') = 'Yes'                                      
      AND ISNULL(AB.Latitude,'') <> ''                                       
      AND ISNULL(AB.Longitude,'') <> ''                                       
      AND ISNULL(AB.ContactPhone,'') <> ''                                  
      AND ISNULL(AB.BVNlatestDate,'') <> ''    
      AND ISNULL(AB.ImageCaptured,0) = 1  
      --AND ISNULL(AB.IsLocationSubmitted,0) = 'Yes'  
      --AND ISNULL(AB.LastName,'') <> ''      
      --AND ISNULL(AB.Businessname,'') <> ''  --AND ISNULL(AB.IsBVNVerified,0) <> 0        
      THEN 1 ELSE 2 END AS KYC_Capture_Status,  
  ISNULL(pos.hasPOS,0) AS hasPOS ,  
  ISNULL(vas.hasVAS, 0) AS hasVAS,  
  ISNULL(bnpl.hasBNPL,0) AS hasBNPL,  
  lastDelvDate,  
  CASE WHEN lastDelvDate IS NULL THEN 0 ELSE 1 END AS isActive  
 INTO #FinalCustomerKYCInfo  
 FROM #FinalTable ab    
 LEFT JOIN LatestUpdateDate lt ON lt.CustomerID = ab.CustomerID  
 LEFT JOIN #dimHasPOS pos ON pos.CustomerID = ab.CustomerID  
 LEFT JOIN #dimHasVAS vas ON vas.CustomerID = ab.CustomerID  
 LEFT JOIN #dimHasBNPL bnpl ON bnpl.CustomerID = ab.CustomerID  
 LEFT JOIN  #dimDevl  delv ON delv.CustomerID = ab.CustomerID;  
   
  
   
    
 ------------------------------------------------------------------------------------  
 SELECT * FROM #FinalCustomerKYCInfo   
  
 ------------------------------------------------------------------------------------  
 DROP TABLE #dimHasPOS;  
 DROP TABLE #dimHasVAS;  
 DROP TABLE #dimHasBNPL;   
 DROP TABLE #ImageUpdate;  
 DROP TABLE #BVNPhonemismatch;  
 DROP TABLE #BVNstatus;  
 DROP TABLE #POCcustomerList;  
 DROP TABLE #CustomerListVerficationStatus;  
 DROP TABLE #DimCustRefID;  
 DROP TABLE #agency;  

  
   
 DROP TABLE #FinalTable;  
 DROP TABLE #FinalCustomerKYCInfo;  
  
 END;   