

SELECT  
    CustomerID as Customer_ID,
    BusinessID  AS Business_ID,
    CAST(CustomerCreatedDate as date) as Created_Date,
    ContactName as Contact_Name,
    ContactPhone as Contact_Phone,
    Statename as State_Name,
    TownName as Town_Name,
    CityName as City_Name,
    Latitude,
    Longitude,
    CustomerStatus as Customer_Status,
    Status,
    FirstName as First_Name, 
    ISNULL(IsLocationCaptured, 0) AS Is_Location_Captured,
    ISNULL(IsLocationSubmitted, 0) AS Is_Location_Submitted,
    LocationSubmittedDate as Location_Submitted_Date,
    ISNULL(IsLocationVerified, 0) AS Is_Location_Verified,
    LocationVerifiedDate as Location_Verified_Date,
    Location,
    FullAddress as Full_Address,
    KYC_Capture_Status, 
    AgentID as Agent_ID,
    AgentName as Agent_Name
FROM VConnectMasterDWR..CustomerKYCdump WITH (NOLOCK)
WHERE BusinessID = 76; 




-- SELECT
--   AF.*
-- FROM (
--    SELECT
--         ContentID as Content_ID,
--         ContactID as Contact_ID,
--         CustomerID as Customer_ID,
--         BusinessID as Business_ID,
--         Createddate as Created_Date,
--         ContactName as Contact_Name,
--         ContactPhone as Contact_Phone,
--         Statename as State_Name,
--         TownName as Town_Name,
--         CityName as City_Name,
--         Latitude,
--         Longitude,
--         CustomerStatus as Customer_Status,
--         Status,
--         FirstName as First_Name,
--         ISNULL(IsLocationCaptured, 0) AS Is_Location_Captured,
--         ISNULL(IsLocationSubmitted, 0) AS Is_Location_Submitted,
--         LocationSubmittedDate as Location_Submitted_Date,
--         ISNULL(IsLocationVerified, 0) AS Is_Location_Verified,
--         LocationVerifiedDate as Location_Verified_Date,
--         Location,
--         FullAddress as Full_Address,
--         ROW_NUMBER() OVER (PARTITION BY AB.CustomerID ORDER BY ContactID DESC) AS rn_active
--     FROM OmniDR..AddressBook AB WITH (NOLOCK)
--     WHERE
--     AB.BusinessID = 76
--     AND ISNULL(Status, 0) IN (0, 1, 2, 7)
-- ) AF
-- WHERE AF.rn_active = 1;
