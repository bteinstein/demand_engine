SELECT
    Stock_Point_ID, 
    CustomerID, 
    MIN(deliverydate) AS first_delv_date,
    MAX(deliverydate) AS last_delv_date
FROM (
    SELECT 
        BusinessID as Stock_Point_ID, CustomerID, deliverydate
    FROM VCONNECTMASTERDWR..tblmanudashsales 
    WHERE Central_BusinessID = 76 
        -- AND BusinessID  = 1647113 -- TESTING WITH CAUSEWAY
        AND deliverydate >= '2025-01-01'
    ) A
GROUP BY Stock_Point_ID, CustomerID