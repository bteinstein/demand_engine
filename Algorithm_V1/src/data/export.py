import os
import pandas as pd

def export_data(
        selected_trip,
        all_push_recommendation,
        cluster_summary,
        stock_point_name,
        CURRENT_DATE,
        LOCAL_EXCEL_PATH
    ): 
    # dir_path = f'./recommendation_output/{CURRENT_DATE}'
    
    # Ensure directory exists
    # os.makedirs(LOCAL_EXCEL_PATH, exist_ok=True)
    
    file_path = f'{LOCAL_EXCEL_PATH}/{stock_point_name}_{CURRENT_DATE}.xlsx'

    with pd.ExcelWriter(file_path) as writer:
        selected_trip.to_excel(writer, sheet_name='Selected Trip', index=False)
        all_push_recommendation.to_excel(writer, sheet_name='All Recommendation', index=False)
        cluster_summary.to_excel(writer, sheet_name='Recommendation Cluster Summary', index=False)
