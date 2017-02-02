import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import time
from matplotlib.dates import date2num
import math

from sklearn.cluster import KMeans

# Method for converting the datetime to seconds since epoch.
def datestr2secs(s):
    return (datetime.strptime(s, '%Y-%m-%d %H:%M:%S')-datetime(1970,1,1)).total_seconds()

def secs2datestr(s):
    return (datetime(1970, 1, 1) + timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S')

# Read in the taxi data.
taxi_sheet = pd.read_csv('green_tripdata_2016-06.csv')

# Take only a day's worth of data.
taxi_sheet.drop(taxi_sheet[taxi_sheet['lpep_pickup_datetime'] < '2016-06-02 00:00:00'].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['lpep_pickup_datetime'] >= '2016-06-03 00:00:00'].index, inplace=True)

# Convert the date time strings to seconds since epoch.
taxi_sheet['lpep_pickup_datetime'] = taxi_sheet['lpep_pickup_datetime'].apply(datestr2secs)
taxi_sheet['Lpep_dropoff_datetime'] = taxi_sheet['Lpep_dropoff_datetime'].apply(datestr2secs)

# Drop fields which are either always constants or meaningless.
taxi_sheet.drop('Store_and_fwd_flag', axis=1, inplace=True)
taxi_sheet.drop('Extra', axis=1, inplace=True)
taxi_sheet.drop('MTA_tax', axis=1, inplace=True)
taxi_sheet.drop('Ehail_fee', axis=1, inplace=True)
taxi_sheet.drop('improvement_surcharge', axis=1, inplace=True)
taxi_sheet.drop('Total_amount', axis=1, inplace=True)
taxi_sheet.drop('VendorID', axis=1, inplace=True)

# Filter out data with no coordinates.
taxi_sheet.drop(taxi_sheet[taxi_sheet['Pickup_longitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Pickup_latitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Dropoff_longitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Dropoff_latitude'] == 0].index, inplace=True)

# Obtain the numpy array from the pandas dataframe
taxi_sheet_data = taxi_sheet.as_matrix()

# Perform the kmeans clustering on the resultant array.
t0 = time()
km = KMeans(n_clusters = 5000)
name = 'kmeans++'
km = km.fit(taxi_sheet_data)
print(time() - t0)

centroids = km.cluster_centers_
labels = km.labels_
results = pd.DataFrame(centroids, columns=list(taxi_sheet.columns.values))

# Convert the date time strings to seconds since epoch.
results['lpep_pickup_datetime'] = results['lpep_pickup_datetime'].apply(secs2datestr)
results['Lpep_dropoff_datetime'] = results['Lpep_dropoff_datetime'].apply(secs2datestr)

print(results)
results.to_csv('results.csv')