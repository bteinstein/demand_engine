
SELECT           
  AF.*          
FROM (          
   SELECT           
        ContentID,          
        ContactID,
        CustomerID,          
        BusinessID,          
        Createddate,          
        ContactName,          
        ContactPhone,          
        Statename,          
        TownName,          
        CityName,          
        Latitude,          
        Longitude,          
        CustomerStatus,
        Status,
        FirstName,             
        ISNULL( IsLocationCaptured, 0) AS IsLocationCaptured ,           
        ISNULL( IsLocationSubmitted, 0) AS IsLocationSubmitted,
        LocationSubmittedDate,
        ISNULL( IsLocationVerified, 0) AS IsLocationVerified ,
        LocationVerifiedDate, 
        Location,
        FullAddress ,  
        --ROW_NUMBER() OVER (PARTITION BY AB.CustomerID ORDER BY contentid ASC) AS rn_onboard,          
        ROW_NUMBER() OVER (PARTITION BY AB.CustomerID ORDER BY ContactID DESC) AS rn_active          
    FROM OmniDR..AddressBook AB WITH (NOLOCK)                  
    WHERE           
    AB.BusinessID = 76     
    AND ISNULL(Status,0) IN (0,1,2,7)        
    --   AND ISNULL(Status,0) = 1     --- VERIFIED CUSTOMER I.E. Approved   
) AF WHERE AF.rn_active = 1 ---(AF.rn_onboard = 1 OR AF.rn_active = 1) 
;      