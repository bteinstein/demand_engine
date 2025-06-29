


USE VCONNECTMASTERDWR;

SELECT TOP 1 * 
	FROM CustomerSKUSalesRecommendationLog WITH (NOLOCK)
	WHERE [status] = 'Active'


EXEC [getCustomerSKURecommendationChecks]


select top 100 * from pipeline_master