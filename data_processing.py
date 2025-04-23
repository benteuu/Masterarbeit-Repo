import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
import matplotlib.pyplot as plt
from geopy.distance import geodesic  # For distance calculation
from math import atan2, degrees, sin, cos, sqrt, radians

#Function for reading data from .plt files
def read_plt(plt_file):
    """
    Read trajectory data from a .plt file into a DataFrame.

    Parameters:
    - plt_file (str): Path to the .plt file to read.

    Returns:
    - pd.DataFrame: Contains columns:
        lat (float): Latitude
        lon (float): Longitude  
        alt (float): Altitude
        elapsed time (float): Time since start
        time (datetime): Combined datetime from date/time columns
        trajectory (str): Filename of source .plt file
        label (int): Transportation mode label (0 if unlabeled)
    """
    #print(f"Processing file: {plt_file}")
     # Determine if the file has a header
    with open(plt_file, "r") as file: first_line = file.readline().strip()
    #print (first_line[0])
    has_header = not first_line[0].isdigit()   #auslesen ob es einen header gibt
    #print(has_header)
    points = pd.read_csv(plt_file,
                        skiprows=1 if has_header else 0,     #erste Zeile überspringen wenn es einen header gibt
                        header=None,        #hier war auf 6 wegen dem was in den orginal datensatz alles da steht
                        #parse=[[5, 6]]  #veraltet
                        #infer_datetime_format=True)  #veraltet
    )
    # Combine columns 5 (date) and 6 (time) into a single 'time' column
    points['time'] = pd.to_datetime(
        points[5] + " " + points[6],  # Concatenate date and time columns
        format='%Y-%m-%d %H:%M:%S',  # Specify the datetime format
        errors='coerce'  # Handle invalid datetime formats gracefully
    )


    #add column with plt number
    points['trajectory']=os.path.basename(plt_file)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt', 4:'elapsed time', 7: 'label'})  #hier gibt es einen konflikt wenn man unten das label lesen nicht ausklammert
    
    # remove unused columns
    points.drop(inplace=True, columns=[2,5,6])
    #print(points)

    return points

def read_plt_full(plt_file):
    """
    Read .plt files with fixed 6-line header structure.

    Parameters:
    - plt_file (str): Path to the .plt file

    Returns:
    - pd.DataFrame: Same structure as read_plt() output
    """
    points = pd.read_csv(plt_file,   
                        skiprows=6,     #erste Zeile überspringen wenn es einen header gibt
                        header=None        #hier war auf 6 wegen dem was in den orginal datensatz alles da steht

    )

    # Combine columns 5 (date) and 6 (time) into a single 'time' column
    points['time'] = pd.to_datetime(
        points[5] + " " + points[6],  # Concatenate date and time columns
        format='%Y-%m-%d %H:%M:%S',  # Specify the datetime format
        errors='coerce'  # Handle invalid datetime formats gracefully
    )


    #add column with plt number
    points['trajectory']=os.path.basename(plt_file)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt', 4:'elapsed time', 7: 'label'})  #hier gibt es einen konflikt wenn man unten das label lesen nicht ausklammert
    
    # remove unused columns
    points.drop(inplace=True, columns=[2,5,6])
    #print(points)

    return points



#If there is a labels file, extract the labels

def read_labels(labels_file):
    """
    Read transportation mode labels from labels.txt file.

    Parameters:
    - labels_file (str): Path to labels.txt file

    Returns:
    - pd.DataFrame: Contains columns:
        label (str): Transportation mode  
        start_time (datetime): Label start time
        end_time (datetime): Label end time
    """
    labels = pd.read_csv(labels_file, skiprows=1, header=None, sep="\t"
                         #parse_dates=[[0, 1], [2, 3]],
                         #infer_datetime_format=True
                         )
    # Combine date and time columns into a single datetime column
    labels['start_time'] = pd.to_datetime(labels[0], format='%Y/%m/%d %H:%M:%S')
    labels['end_time'] = pd.to_datetime(labels[1], format='%Y/%m/%d %H:%M:%S')
    # Drop the original date and time columns
    labels.drop(inplace=True, columns=[0, 1])
    
    #for clarity rename columns
    labels.columns = ['label','start_time', 'end_time']
    
    return labels


def apply_labels(points, labels):
    """
    Assign transportation labels to GPS points based on time intervals.

    Parameters:
    - points (pd.DataFrame): DataFrame from read_plt() 
    - labels (pd.DataFrame): DataFrame from read_labels()

    Modifies:
    - Adds 'label' column to points DataFrame (0 for unlabeled)
    """
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1    
    
    no_label_condition = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points.loc[no_label_condition, 'label'] = 0



#Put together data for all the users

def read_user(user_folder):
    labels = None
    if os.path.exists(os.path.join(user_folder, 'Trajectory')):   #this is how it is differntiated between the full or partial dataset
        plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
        df = pd.concat([read_plt_full(f) for f in plt_files])

        labels_file = os.path.join(user_folder, 'labels.txt')
        if os.path.exists(labels_file):
            labels = read_labels(labels_file)
            apply_labels(df, labels)
        else:
            df['label'] = 0
        return df
    else:
        plt_files = glob.glob(os.path.join(user_folder, '*.plt'))    
        df = pd.concat([read_plt(f) for f in plt_files])
    return df


#Read data from all users

def read_all_users(folder):
    """
    Process trajectory data for all users in a directory.

    Parameters:
    - folder (str): Path to directory containing user folders

    Returns:
    - pd.DataFrame: Combined data from all users with:
        user (int): User ID
        (other columns same as read_plt())
    """
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('.')] #damit ds._store nicht gemacht wird
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder,sf))
        df['user'] = int(sf.replace("User", ""))  #hier änderung weil "user" bei ihrem datensatz davor steht
        dfs.append(df)
        df['time'] = pd.to_datetime(df['time'])  #transform data to datetime
    return pd.concat(dfs)

def angular_difference(prev_bearing, curr_bearing):
    """
    Calculate smallest angular difference between two bearings.

    Parameters:
    - prev_bearing (float): Previous bearing in degrees (0-360)
    - curr_bearing (float): Current bearing in degrees (0-360)

    Returns:
    - float: Angular difference in degrees (0-180)
    """
    diff = np.abs(curr_bearing - prev_bearing)
    return np.minimum(diff, 360 - diff)


def calculations(df, segment='segment'):
    """
    Calculate movement features for trajectory segments.

    Parameters:
    - df (pd.DataFrame): Input trajectory data
    - segment (str): Column name for segmentation

    Returns:
    - pd.DataFrame: Original data with added features:
        distance (float): km between consecutive points
        speed (float): km/h 
        acceleration (float): km/h²
        jerk (float): Rate of acceleration change
        bearing (float): Movement direction (0-360°)
        heading_change (float): Degrees from previous bearing
        angular_velocity (float): Degrees/sec
        angular_acceleration (float): Degrees/sec²
        Vrate (float): Absolute speed change ratio
    """
    df = df.sort_values(by=[segment, 'time'])
    #initialize
    df['distance'] = 0.0  # Distance between consecutive points
    df['speed'] = 0.0     # Speed between consecutive points

    #Function to calculate the distance
    def calculate_distance(lat1, lon1, lat2, lon2):

        # Convert coordinates to radians using NumPy vectorize
        lat1_rad = np.deg2rad(lat1)
        lon1_rad = np.deg2rad(lon1)
        lat2_rad = np.deg2rad(lat2)
        lon2_rad = np.deg2rad(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = 6371 * c             #radius of earth

        return distance


    #Function to calculate Bearing
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """
        Calculate the bearing between two geographic points.
        Parameters:
        lat1, lon1: Latitude and Longitude of Point 1 (in degrees)
        lat2, lon2: Latitude and Longitude of Point 2 (in degrees)
    
        Returns:
        Bearing (in degrees)
        """
        #Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Calculate the difference in longitude
        delta_lon = lon2 - lon1

        # Compute the bearing
        y = np.sin(delta_lon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        bearing = np.arctan2(y, x)

        # Convert bearing from radians to degrees and normalize to 0-360
        bearing = (np.degrees(bearing) + 360) % 360

        return bearing
    
    #Function to apply all the calculations by segment
    def calculate_segment(group):
        group['time_diff'] = group['time'].diff().dt.total_seconds()

        group['distance'] = calculate_distance(group['lat'].shift().values, group['lon'].shift().values, group['lat'].values, group['lon'].values) #shift nimmt immer den vorherigen?

        group['speed'] = group['distance'] / (group['time_diff'] / 3600)  # km/h

        group['acceleration'] = group['speed'].diff() / group['time_diff']

        # Compute jerk (rate of change of acceleration)
        group['jerk'] = group['acceleration'].diff() / group['time_diff']
    
        group['bearing'] = calculate_bearing(group['lat'].shift().values, group['lon'].shift().values, group['lat'].values, group['lon'].values) #shift nimmt immer den vorherigen?

        #ist heading chamge so richtig??
        group['heading_change'] = group.bearing.diff().abs()

        group['Vrate'] = np.abs(group['speed'].diff()) / group['speed']


        #  Heading change (smallest angular difference)
        bearing_diff = angular_difference(group['bearing'].shift(), group['bearing'])
        group['heading_change'] = bearing_diff
        
        #  Angular velocity (smallest diff / time)
        group['angular_velocity'] = bearing_diff / group['time_diff']

        # Compute angular acceleration (change in angular velocity per second)
        group['angular_acceleration'] = group['angular_velocity'].diff() / group['time_diff']

        return group
    
    #Now process each segment individually
    df = df.groupby(segment, group_keys=False).apply(calculate_segment)

    #  Drop infinite and na values from the dataset
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df



#add 'taxi' to 'car' and 'subway' to 'train' to merge similar classes
# leave out 'run', 'boat', 'airplane', 'motorcycle (small sample size)
def process_classes(df):
    """
    Merge similar transportation classes and filter rare modes.

    Parameters:
    - df (pd.DataFrame): Input data with labels

    Returns:
    - pd.DataFrame: Data with modified labels:
        Merges: taxi->car, subway->train
        Removes: boat, run, airplane, motorcycle
    """
    # Merge classes with similar mobility patterns
    df['label'] = df['label'].replace({
        'taxi': 'car',
        'subway': 'train'
    })

    #leave out classes with very small sample sizes
    # Exclude data points labeled as 'boat'
    df = df[df['label'] != 'boat']
    # Exclude data points labeled as 'run'
    df = df[df['label'] != 'run']
    # Exclude data points labeled as 'airplane'
    df = df[df['label'] != 'airplane']
    # Exclude data points labeled as 'motorcycle'
    df = df[df['label'] != 'motorcycle']
    
    return df



def create_segments(df):
    """
    Split trajectories into segments based on label/time changes.

    Parameters:
    - df (pd.DataFrame): Input trajectory data

    Returns:
    - pd.DataFrame: Data with new 'segment' column grouping:
        - Same trajectory
        - Same transportation label  
        - Max 3 minutes between points
    """
    df = df.sort_values(by=['trajectory', 'time'])
    # Initialize the segment number
    segment_number = 0
    # Create a new column for segment numbers
    df['segment'] = 0

    # Create a boolean mask where the trajectory name or the label changes or more than 20mins pass

    mask = (
        (df['trajectory'] != df['trajectory'].shift()) |  # Trajectory change
        (df['label'] != df['label'].shift()) |  # Label change
        (df['time'].diff().dt.total_seconds() > 180)  # More than 3 minutes (180 sec)
    )
    df['segment'] = mask.cumsum()

    return df



def drop_unlabelled(df):
    """
    Filter out unlabeled data points (label=0).

    Parameters:
    - df (pd.DataFrame): Input data

    Returns:
    - pd.DataFrame: Data without unlabeled points
    """
    # Drop data points that are not labeled
    df = df[df['label'] != 0]
    return df


#Get all unlabeled trajectories
def get_unlabelled(df):
    """
    Get only unlabeled data points (label=0).

    Parameters:
    - df (pd.DataFrame): Input data

    Returns:
    - pd.DataFrame: Only unlabeled points
    """
    df = df[df['label'] == 0]
    return df





def filter(df, max_speed_kmh=250, max_distance_km=100, 
          mode_thresholds={'walk': 25, 'bike': 60, 'car': 200, 'train': 300, 'bus': 200, '0': 300},
          segment_col='segment', label_col='label'):
    """
    Filter out invalid points and their next two points within segments
    
    Parameters:
    - df: Input DataFrame with speed in km/h and distance in km
    - max_speed_kmh: Default speed threshold for unspecified labels
    - max_distance_km: Maximum allowed distance between consecutive points
    - mode_thresholds: Dictionary of {label: max_speed_kmh} for specific modes
    - segment_col: Column name for segment identifiers
    - label_col: Column name for transportation mode labels
    
    Returns:
    - Filtered DataFrame with invalid points and their next two removed
    """
    df = df.reset_index(drop=True)
    indices_to_remove = set()
    
    # Process each segment independently
    for _, segment in df.groupby(segment_col):
        if segment.empty:
            continue
            
        # Get segment's transportation mode from first row
        label = segment[label_col].iloc[0]
        speed_threshold = mode_thresholds.get(label, max_speed_kmh)
        
        # Get ordered indices and process sequentially
        seg_indices = segment.index.tolist()
        n = len(seg_indices)
        i = 0
        
        while i < n:
            current_idx = seg_indices[i]
            current_row = segment.loc[current_idx]
            
            # Check if current point violates thresholds
            if (current_row['speed'] > speed_threshold) or \
               (current_row['distance'] > max_distance_km):
                
                # Mark current + next two indices for removal
                for offset in range(3):
                    if i + offset < n:
                        indices_to_remove.add(seg_indices[i + offset])
                
                # Skip ahead by 3 positions
                i += 2
            else:
                i += 1
    
    return df[~df.index.isin(indices_to_remove)].copy()

