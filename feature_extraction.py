#Extract features

import pandas as pd
import numpy as np

#Grouped for every trajectory
def extract_features(group):
    features = {}
    
    features['label'] = group['label'].iloc[0]
    features['segment'] = group['segment'].iloc[0]
    # Trajectory-level information
    features['trajectory'] = group['trajectory'].iloc[0]
    features['start_time'] = group['time'].iloc[0].hour  # Extract the hour of the day
    features['trajectory_length'] = len(group)  # Total number of points
    
    # Distance-based features
    features['total_distance'] = group['distance'].sum()
    features['expected_speed'] = group['speed'].mean()
    features['max_speed'] = group['speed'].max()
    #expectation of speed for analyzing distributions of speed samples
    # Mean Speed
    total_time = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds() / 3600  # Convert to hours
    features['mean_speed'] = features['total_distance'] / total_time if total_time > 0 else 0  # Avoid division by zero

    # Top 3 Speeds#
    sorted_speeds = np.sort(group['speed'].dropna().values)[::-1]  # Sort in descending order
    features['top_1_speed'] = sorted_speeds[0] if len(sorted_speeds) > 0 else 0
    features['top_2_speed'] = sorted_speeds[1] if len(sorted_speeds) > 1 else 0
    features['top_3_speed'] = sorted_speeds[2] if len(sorted_speeds) > 2 else 0

    # Top 3 Accelerations
    sorted_accelerations = np.sort(group['acceleration'].dropna().values)[::-1]  # Sort in descending order
    features['top_1_acceleration'] = sorted_accelerations[0] if len(sorted_accelerations) > 0 else 0
    features['top_2_acceleration'] = sorted_accelerations[1] if len(sorted_accelerations) > 1 else 0
    features['top_3_acceleration'] = sorted_accelerations[2] if len(sorted_accelerations) > 2 else 0

    
    # Time-based features
    duration = (group['time'].iloc[-1] - group['time'].iloc[0]).total_seconds() / 3600  # Duration in hours
    features['duration'] = duration
    features['average_acceleration'] = group['acceleration'].mean()
    
    # Variability features
    features['speed_std'] = group['speed'].std()
    features['acceleration_std'] = group['acceleration'].std()
    
    # Heading change rate (HCR)
    features['bearing_changes'] = (group['heading_change']>19).sum()
    features['HCR'] = features['bearing_changes']/features['trajectory_length']

    # Stop Rate (VCR)
    features['stops'] = (group['speed']<0.000277778).sum()    #Im paper: 3.4. Aber hier ist Kilometer/Sekunde und im Paper Kilometer/Stunde daher 0.009. aber der sorgt dafür, dass walken immer unetrm treshhold ist
    features['SR'] = features['stops']/features['trajectory_length']

    #Velocity Change Tare (VCR
    features['VC'] = (group['Vrate'] > 0.26).sum()
    features['VCR'] = features['VC']/features['trajectory_length']

    return pd.Series(features)


def clean_features (features):
    #Replace Infinite values with NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    #Drop columns with NaN values
    features = features.dropna(axis=0)

    return features

#noch ne wrapper function für beide? oder auch für komplette datenverarbeitung