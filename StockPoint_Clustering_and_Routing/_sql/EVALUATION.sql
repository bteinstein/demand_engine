 USE VConnectMasterDWR;

 EXEC SP_HELPTEXT Usp_Generate_Agent_Payment_By_Customer_Score 

select count(*) from VConnectMasterDWR.gis_analysis.stockpoint_h3_coverage WITH (NOLOCK)
select top 10 * from VConnectMasterDWR.gis_analysis.stockpoint_h3_coverage WITH (NOLOCK)

select top 10 * from VConnectMasterDWR.gis_analysis.h3_cells WITH (NOLOCK) WHERE h3_cell = '88589c996bfffff'
select top 10 * from VConnectMasterDWR.gis_analysis.stockpoint_h3_coverage WITH (NOLOCK)
select top 10 * from VConnectMasterDWR.gis_analysis.customer_stockpoint_cluster_assignment WITH (NOLOCK)


select top 10 * from VConnectMasterDWR.gis_analysis.h3_cells WITH (NOLOCK)
select count(*) from VConnectMasterDWR.gis_analysis.h3_cells WITH (NOLOCK)
select count(distinct h3_cell) from VConnectMasterDWR.gis_analysis.h3_cells WITH (NOLOCK)



select top 10 * from VConnectMasterDWR.gis_analysis.customer_stockpoint_cluster_assignment WITH (NOLOCK)
select count(*) from VConnectMasterDWR.gis_analysis.customer_stockpoint_cluster_assignment WITH (NOLOCK)


h3_cell
88581354c1fffff

--DECLARE @sql NVARCHAR(MAX);
--DECLARE @schema_name NVARCHAR(255) = 'migrated_data';

---- Generate DROP statements for all tables in the schema
--SELECT @sql = STRING_AGG(CONCAT('DROP TABLE [', SCHEMA_NAME(schema_id), '].[' , name, '];'), CHAR(10))
--FROM sys.tables
--WHERE schema_id = SCHEMA_ID(@schema_name);

--PRINT(@sql)
---- Execute the generated SQL to drop all tables
--EXEC sp_executesql @sql;

---- Now try dropping the schema
--BEGIN TRY
--    EXEC('DROP SCHEMA IF EXISTS ' + @schema_name);
--    PRINT 'Schema dropped successfully';
--END TRY
--BEGIN CATCH
--    PRINT 'Error dropping schema: ' + ERROR_MESSAGE();
--END CATCH
