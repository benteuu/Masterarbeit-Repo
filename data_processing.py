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


#If there is a labels file, extract the labels

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

   #for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

   #replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels


#Put together data for all the users

def read_user(user_folder):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, '*.plt'))    #'Trajectory' folder raus
    df = pd.concat([read_plt(f) for f in plt_files])

    #labels_file = os.path.join(user_folder, 'labels.txt')    #ich glaub das hier ist alles unnötig weil ich nur welche mit labels hab
    #if os.path.exists(labels_file):
    #    labels = read_labels(labels_file)
    #    apply_labels(df, labels)
    #else:
    #    df['label'] = 0
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
    return df


def plusfour(c):
    a=c+4
    return a