-- SQL SERVER
WITH ranked_cells AS (
    SELECT
        h3_cell,
        CASE
            WHEN ward_name IS NULL OR lga_name IS NULL OR state_name IS NULL THEN NULL
            ELSE 'NG' + ' | ' +
                     TRIM(UPPER(state_name)) + ' | ' +
                     COALESCE(NULLIF(TRIM(UPPER(REPLACE(REPLACE(REPLACE(TRIM(lga_name), ' / ', '/'), 'Unknown', ''), '- ', '-'))), ''), '-') + ' | ' +
                     COALESCE(NULLIF(TRIM(UPPER(REPLACE(REPLACE(REPLACE(TRIM(ward_name), ' / ', '/'), 'Unknown', ''), '- ', '-'))), ''), '-') + '-' +
                     CAST(ROW_NUMBER() OVER (
                         PARTITION BY
                             state_name,
                             CASE WHEN lga_name IS NULL THEN NULL ELSE REPLACE(REPLACE(REPLACE(TRIM(lga_name), ' / ', '/'), 'Unknown', ''), '- ', '-') END,
                             CASE WHEN ward_name IS NULL THEN NULL ELSE REPLACE(REPLACE(REPLACE(TRIM(ward_name), ' / ', '/'), 'Unknown', ''), '- ', '-') END
                         ORDER BY h3_cell ASC
                     ) AS VARCHAR)
        END AS new_h3_derived_id
    FROM VConnectMasterDWR.gis_analysis.h3_cells
    -- WHERE state_name = 'Lagos'
)
UPDATE T1
SET h3_derived_id = T2.new_h3_derived_id
FROM VConnectMasterDWR.gis_analysis.h3_cells AS T1
JOIN ranked_cells AS T2 ON T1.h3_cell = T2.h3_cell;