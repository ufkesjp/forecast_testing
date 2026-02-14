import pandas as pd
import numpy as np


# ---- Phase 1: DIAGNOSTIC FUNCTIONS ----

def get_diagnostics(df):
    def calculate_metrics(group):
        series = group['quantity_field']
        demand_periods = np.count_nonzero(series)
        if demand_periods == 0:
            adi = np.nan
            cv2 = np.nan
            pattern = "No Demand"
        else:
            adi = len(series) / demand_periods
            nonzero_demand = series[series > 0]
            cv2 = (nonzero_demand.std() / nonzero_demand.mean())**2 if nonzero_demand.mean() != 0 else 0
            if adi < 1.32 and cv2 < 0.49:
                pattern = "Smooth"
            elif adi < 1.32 and cv2 >= 0.49:
                pattern = "Erratic"
            elif adi >= 1.32 and cv2 < 0.49:
                pattern = "Intermittent"
            else:                pattern = "Lumpy"
        return pd.Series({
            'ADI': adi, 
            'CV2': cv2, 
            'Demand Pattern': pattern
        })
    
    def detect_outliers(group):
        series = group['quantity_field']
        dates = group['date_field']
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        mask = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        return ", ".join(dates[mask].dt.strftime('%Y-%m-%d').tolist())
    
    # calculate metrics per group, selecting only necessary columns
    metrics = (
        df.groupby('item_field')[['quantity_field']]
        .apply(calculate_metrics)
        .reset_index()
    )

    # detect outliers per group, selecting only necessary columns
    outliers = (
        df.groupby('item_field')[['quantity_field', 'date_field']]
        .apply(detect_outliers)
        .reset_index()
    )
    outliers.columns = ['item_field', 'Flagged Outliers']

    # merge metrics and outliers
    diagnostics = pd.merge(metrics, outliers, on= 'item_field', how='left')
    diagnostics = diagnostics.rename(columns={'item_field': 'Forecast Series'})

    # return the desired columns 
    return diagnostics[['Forecast Series', 'ADI', 'CV2', 'Demand Pattern', 'Flagged Outliers']]

    
