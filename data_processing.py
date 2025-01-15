import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
import matplotlib.pyplot as plt
from geopy.distance import geodesic  # For distance calculation



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

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

   #for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

   #replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels

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

def read_all_users(folder):
    subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('.')] #damit ds._store nicht gemacht wird
    dfs = []
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i + 1, len(subfolders), sf))
        df = read_user(os.path.join(folder,sf))
        df['user'] = int(sf.replace("User", ""))  #hier änderung weil "user" bei ihrem datensatz davor steht
        dfs.append(df)
        df['time'] = pd.to_datetime(df['time'])
    return pd.concat(dfs)