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
    #print(f"Processing file: {plt_file}")
     # Determine if the file has a header
    #with open(plt_file, "r") as file: first_line = file.readline().strip()
    #print (first_line[0])
    #has_header = not first_line[0].isdigit()   #auslesen ob es einen header gibt
    #print(has_header)
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
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('.')] #damit ds._store nicht gemacht wird
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder,sf))
        df['user'] = int(sf.replace("User", ""))  #hier änderung weil "user" bei ihrem datensatz davor steht
        dfs.append(df)
        df['time'] = pd.to_datetime(df['time'])  #transform data to datetime
    return pd.concat(dfs)


def calculations(df):
    df = df.sort_values(by=['trajectory', 'time'])
    #initialize
    df['distance'] = 0.0  # Distance between consecutive points
    df['speed'] = 0.0     # Speed between consecutive points

    df['time_diff'] = df['time'].diff().dt.total_seconds()

    #Calculate the distance
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
    
    df['distance'] = calculate_distance(df['lat'].shift().values, df['lon'].shift().values, df['lat'].values, df['lon'].values) #shift nimmt immer den vorherigen?

    df['speed'] = df['distance'] / df['time_diff']

    df['acceleration'] = df['speed'].diff() / df['time_diff']

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
    
    df['bearing'] = calculate_bearing(df['lat'].shift().values, df['lon'].shift().values, df['lat'].values, df['lon'].values) #shift nimmt immer den vorherigen?

    df['heading_change'] = df.bearing.diff().abs()

    df['Vrate'] = np.abs(df['speed'].diff()) / df['speed']

    # Compute angular velocity (change in bearing per second)
    df['angular_velocity'] = df['bearing'].diff() / df['time_diff']

    # Compute angular acceleration (change in angular velocity per second)
    df['angular_acceleration'] = df['angular_velocity'].diff() / df['time_diff']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)



    return df



#add 'taxi' to 'car' and 'subway' to 'train' to merge similar classes
# leave out 'run', 'boat', 'airplane', 'motorcycle (small sample size)
def process_classes(df):
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

#In the large dataset, not every trajectory has the same label. OR more than 20mins passed!
#Thus, the trajectories have to be split in segments with the same label
def create_segments(df):
    # Initialize the segment number
    segment_number = 0
    # Create a new column for segment numbers
    df['segment'] = 0

    # Create a boolean mask where the trajectory name or the label changes
    mask = (df['trajectory'] != df['trajectory'].shift()) | (df['label'] != df['label'].shift())

    # Use cumsum to increment the segment number where the mask is True
    df['segment'] = mask.cumsum()

    return df



def drop_unlabelled(df):
    # Drop data points that are not labeled
    df = df[df['label'] != 0]
    return df

#def get_unlabelled(df):
    