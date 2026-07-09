# disaggregation_utility.py
import numpy as np
import pandas as pd

def apply_bounded_disaggregation(df):
    """
    OPTION A IMPLEMENTATION:
    Distributes historical division-level affected macro-counts ONLY to pixels
    that were actually impacted (where Affected_People > 0 initially or where
    the environmental criteria show active flooding).
    """
    df = df.copy()

    # Identify the rows that represent active flood locations in the historical data
    # (Pixels where historical ground-truth lists anomalies or high baseline presence)
    # We create a mask for pixels that are part of the active historical footprint
    is_impacted_footprint = (df['Affected_People'] > 0)

    # Calculate total population inside ONLY the flooded footprint per division per year
    df['_flooded_pop_subtotal'] = df['Ghs_Pop_Baseline'] * is_impacted_footprint
    div_flooded_pop_totals = df.groupby(['Ds_Division_Name', 'Data_Year'])['_flooded_pop_subtotal'].transform('sum')

    # Calculate how many flooded pixels exist in each division event
    div_flooded_pixel_counts = df.groupby(['Ds_Division_Name', 'Data_Year'])['_flooded_pop_subtotal'].transform('count')

    # Compute share: if a pixel isn't in the wet zone, its share is 0
    df['_pixel_share'] = np.where(
        is_impacted_footprint & (div_flooded_pop_totals > 0),
        df['Ghs_Pop_Baseline'] / div_flooded_pop_totals,
        0.0
    )

    # Fallback for edge cases where pop totals are zero but entries exist
    df['_pixel_share'] = np.where(
        is_impacted_footprint & (div_flooded_pop_totals == 0),
        1.0 / (div_flooded_pixel_counts + 1e-6),
        df['_pixel_share']
    )

    # Apply bounded allocation
    if 'Affected_People' in df.columns:
        df['Affected_People'] = df['Affected_People'] * df['_pixel_share']
    if 'Affected_Families' in df.columns:
        df['Affected_Families'] = df['Affected_Families'] * df['_pixel_share']

    # Clean up tracking features
    df.drop(columns=['_flooded_pop_subtotal', '_pixel_share'], inplace=True, errors='ignore')
    return df

def compute_live_heuristic_features(df, input_precip_mm):
    """
    Dynamically recalculates DMC-aligned severity and engineered weights
    based on live operational runtime rainfall parameters.
    """
    df = df.copy()

    if input_precip_mm <= 50:
        sev_weight = 0.3
    elif input_precip_mm <= 100:
        sev_weight = 0.6
    else:
        sev_weight = 1.0

    df['Precip_Mm'] = input_precip_mm
    df['Severity_Weight'] = sev_weight

    if 'Ghs_Built_S_Total' in df.columns:
        df['Built_Up_Ratio'] = df['Ghs_Built_S_Total'] / 10000.0
    else:
        df['Built_Up_Ratio'] = 0.0

    def get_occ_adj(row):
        return 1.2 if (row.get('Is_Holiday', 0) == 1 or row.get('Is_Weekend', 0) == 1) else 0.85

    df['Occupancy_Adj'] = df.apply(get_occ_adj, axis=1)

    weighted_calc = (
            df['Ghs_Pop_Baseline'] * df['Built_Up_Ratio'] * df['Severity_Weight'] * df['Occupancy_Adj']
    )
    df['Weighted_Pop_Engineered'] = np.minimum(np.maximum(weighted_calc, 0), df['Ghs_Pop_Baseline'])

    return df