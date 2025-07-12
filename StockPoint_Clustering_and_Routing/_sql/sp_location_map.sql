USE VCONNECTMASTERDWR;

-- DROP TABLE IF EXISTS #SP_LOCATION_MAPPING;
SELECT DISTINCT 
            fm.Fulfilement_Center_ID AS FCID, 
            sp.Stock_point_Name AS StockPointName, 
            sm.StateName, 
            cm.StateID AS State_ID,
            lgm.Region,
            cm.CityName AS LGA, 
            tm.CityID AS LGA_ID,
            tm.TownName AS LCDA, 
            fm.Location_ID AS LCDA_ID 
        -- INTO #SP_LOCATION_MAPPING
	    FROM  FC_Location_Mapping fm WITH (NOLOCK) 
        INNER JOIN  Townmaster tm WITH (NOLOCK) ON fm.location_id = tm.Contentid
        INNER JOIN  citymaster cm WITH (NOLOCK) ON tm.CityID = cm.Contentid
        INNER JOIN  statemaster sm WITH (NOLOCK) ON cm.stateid = sm.Contentid
        LEFT JOIN  LGA_Region_Master lgm WITH (NOLOCK) ON lgm.LGA_ID = tm.CityID
        INNER JOIN  Stock_Point_Master sp WITH (NOLOCK) ON fm.Fulfilement_Center_ID = sp.Stock_Point_ID
        WHERE 
        fm.[status] = 1 -- Active
        AND  fm.Location_Type = 3 -- LCDA
        -- AND  lgm.CountryID = 1
        AND  (Is_Fulfilement_Center = 1 OR Is_Mfc = 1 OR Is_Mfc = 0)
        AND  sp.Fulfilement_Center_ID = 76
        AND sp.Stock_point_Name NOT LIKE '%Test%';