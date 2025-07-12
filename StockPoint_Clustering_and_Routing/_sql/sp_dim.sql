SELECT 
	sp.Stock_Point_ID, sp.Stock_point_Name, 
	bm.Lattitude, bm.Longitude 
FROM Stock_Point_Master sp WITH (NOLOCK)  
INNER JOIN BusinessMaster bm WITH (NOLOCK) ON sp.Stock_Point_ID = bm.Contentid
WHERE sp.Fulfilement_Center_ID = 76
	AND sp.Status = 1
	AND sp.Warehouse_Type IN (1, 3)
	AND sp.Stock_Point_Name NOT LIKE '%TEST%'
	AND sp.Is_Mfc = 1
