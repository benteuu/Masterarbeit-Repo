#Extract features

import pandas as pd
import numpy as np

def extract_features(df, group_column='segment', reset_index=True):
    """
    Compute trajectory features for groups in a DataFrame.
    
    df: Input DataFrame with trajectory data.
    group_column: Column to group by (default: 'segment').
    reset_index: Whether to reset the index of the output (default: True).
    
    Returns:
        DataFrame with extracted features for each group.
    """
    # Check if grouping column exists
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in DataFrame.")
    
    # Define helper function to process a single group
    def _process_group(group):
        features = {}
        
        # Basic metadata
        features['label'] = group['label'].iloc[0]
        features['segment'] = group[group_column].iloc[0]
        features['trajectory'] = group['trajectory'].iloc[0]
        features['start_time'] = group['time'].iloc[0].hour
        
        # Trajectory length and distance
        features['segment_length'] = len(group)
        features['total_distance'] = group['distance'].sum()

        # Time-based features
        duration = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds() / 3600
        features['duration'] = duration
        
        # Speed features
        features['mean_speed'] = features['total_distance'] / duration if duration > 0 else 0
        features['expected_speed'] = group['speed'].mean()
        #features['max_speed'] = group['speed'].max()
        sorted_speeds = np.sort(group['speed'].dropna())[::-1]
        features['top_1_speed'] = sorted_speeds[0] if len(sorted_speeds) > 0 else np.nan
        features['top_2_speed'] = sorted_speeds[1] if len(sorted_speeds) > 1 else np.nan
        features['top_3_speed'] = sorted_speeds[2] if len(sorted_speeds) > 2 else np.nan
        

        # Acceleration features
        features['average_acceleration'] = group['acceleration'].mean()
        sorted_accels = np.sort(group['acceleration'].dropna())[::-1]
        features['top_1_acceleration'] = sorted_accels[0] if len(sorted_accels) > 0 else np.nan
        features['top_2_acceleration'] = sorted_accels[1] if len(sorted_accels) > 1 else np.nan
        features['top_3_acceleration'] = sorted_accels[2] if len(sorted_accels) > 2 else np.nan

        # Jerk-based features
        features['max_jerk'] = group['jerk'].max()
        features['mean_jerk'] = group['jerk'].mean()
        features['jerk_std'] = group['jerk'].std()
    
        # Top 3 jerks
        sorted_jerks = np.sort(group['jerk'].abs().dropna().values)[::-1]
        features['top_1_jerk'] = sorted_jerks[0] if len(sorted_jerks) > 0 else np.nan
        features['top_2_jerk'] = sorted_jerks[1] if len(sorted_jerks) > 1 else np.nan
        features['top_3_jerk'] = sorted_jerks[2] if len(sorted_jerks) > 2 else np.nan
        
        # Variability
        features['speed_std'] = group['speed'].std()
        features['acceleration_std'] = group['acceleration'].std()
        
        # HCR (Heading Change Rate)
        segments = len(group)
        bearing_changes = (group['heading_change'].abs() > 19).sum()  # Threshold: 19 degrees
        features['HCR'] = bearing_changes / segments if segments > 0 else 0
        
        # SR (Stop Rate)
        stops = (group['speed'] < 1).sum()  # Threshold: 1kmh
        features['SR'] = stops / len(group)
        
        # VCR (Velocity Change Rate)
        vc = (group['Vrate'] > 3.37).sum()  # Threshold: 0.26 km/s² (adjust units as needed)
        features['VCR'] = vc / segments if segments > 0 else 0

        features['speed_skewness'] = group['speed'].skew()  
        features['acc_skewness'] = group['acceleration'].skew() 
        features['speed_kurtosis'] = group['speed'].kurtosis()  
        features['acceleration_kurtosis'] = group['acceleration'].kurtosis()  
        features['speed_autocorr'] = group['speed'].autocorr() 
        features['acc_autocorr'] = group['acceleration'].autocorr() 
  
        
        return pd.Series(features)
    
    # Apply grouping and feature extraction
    feature_df = df.groupby(group_column).apply(_process_group)
    return feature_df



def clean_features (features):
    #Replace Infinite values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    #Drop columns with NaN values
    features = features.dropna(axis=0)

    return features
