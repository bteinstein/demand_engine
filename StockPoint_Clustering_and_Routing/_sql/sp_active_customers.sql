SELECT 
    Stock_Point_ID, 
    CustomerID as Customer_ID, 
    MIN(deliverydate) AS first_delv_date,
    MAX(deliverydate) AS last_delv_date
FROM (
    SELECT  
        BusinessID as Stock_Point_ID, CustomerID, deliverydate
    FROM VCONNECTMASTERDWR..tblmanudashsales with (nolock)
    WHERE Central_BusinessID = 76  
	      AND deliverydate >= '2025-01-01'
    ) A
GROUP BY Stock_Point_ID, CustomerID