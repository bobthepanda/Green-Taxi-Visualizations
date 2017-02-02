import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import time
from matplotlib.dates import date2num
import math

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

num_items = 46339

# Method for converting the datetime to seconds since epoch.
def datestr2secs(s):
    return (datetime.strptime(s, '%Y-%m-%d %H:%M:%S')-datetime(1970,1,1)).total_seconds()

def secs2datestr(s):
    return (datetime(1970, 1, 1) + timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S')

# Read in the taxi data.
taxi_sheet = pd.read_csv('../web/csv/green_tripdata_2016-06.csv')

# Take only a day's worth of data.
taxi_sheet.drop(taxi_sheet[taxi_sheet['lpep_pickup_datetime'] < '2016-06-02 00:00:00'].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['lpep_pickup_datetime'] >= '2016-06-03 00:00:00'].index, inplace=True)

taxi_sheet.drop(taxi_sheet[taxi_sheet['Pickup_longitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Pickup_latitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Dropoff_longitude'] == 0].index, inplace=True)
taxi_sheet.drop(taxi_sheet[taxi_sheet['Dropoff_latitude'] == 0].index, inplace=True)


# Convert the date time strings to seconds since epoch.
taxi_sheet['lpep_pickup_datetime'] = taxi_sheet['lpep_pickup_datetime'].apply(datestr2secs)
taxi_sheet['Lpep_dropoff_datetime'] = taxi_sheet['Lpep_dropoff_datetime'].apply(datestr2secs)
taxi_sheet.replace(['N', 'Y'], [0, 1], inplace=True)
taxi_sheet.fillna(0, inplace=True)

# Drop fields which are either always constants or meaningless.
taxi_sheet.drop('Store_and_fwd_flag', axis=1, inplace=True)
taxi_sheet.drop('Extra', axis=1, inplace=True)
taxi_sheet.drop('MTA_tax', axis=1, inplace=True)
taxi_sheet.drop('Ehail_fee', axis=1, inplace=True)
taxi_sheet.drop('improvement_surcharge', axis=1, inplace=True)
taxi_sheet.drop('Total_amount', axis=1, inplace=True)
taxi_sheet.drop('VendorID', axis=1, inplace=True)
taxi_sheet.drop('RateCodeID', axis=1, inplace=True)
taxi_sheet.drop('Trip_type ', axis=1, inplace=True)


# Perform PCA analysis.
numeric_data = taxi_sheet._get_numeric_data()
pca = PCA(n_components=10)
pca.fit(numeric_data)
proj_data = pca.transform(numeric_data)

# Obtain the covariance matrix.
centered_data = numeric_data - np.mean(numeric_data, axis=0)
cov = np.dot(centered_data.T, centered_data) / num_items

# Grab the eigenvalues and eigenvectors.
eigenvals, eigenvecs = np.linalg.eig(cov)

# Calculate the sum of squared loadings.
squared_loadings = []
for i in range(0, len(eigenvecs)):
    squared_loadings.append(np.sum(np.square(eigenvecs[i][:10])))

# Order features by contribution to most significant PCA vectors.
attributes = np.array(taxi_sheet.columns.values.tolist())
squared_loadings = np.array(squared_loadings)
sorter = np.argsort(squared_loadings)
attributes = attributes[sorter][::-1]
squared_loadings = squared_loadings[sorter][::-1]

# Reduce the numpy array to the most important attributes.
taxi_sheet = taxi_sheet[attributes[:10]]
print(taxi_sheet.shape)

# Obtain the numpy array from the pandas dataframe
taxi_sheet_data = taxi_sheet.as_matrix()

# Perform the kmeans clustering on the resultant array.
km = KMeans(n_clusters = 1000)
name = 'kmeans++'
km = km.fit(taxi_sheet_data)

# Obtain the clusters and transfer them to a results csv.
centroids = km.cluster_centers_
km_results = pd.DataFrame(centroids, columns=list(taxi_sheet.columns.values))
print(km_results.shape)

# Convert the date time strings to seconds since epoch.
km_results['lpep_pickup_datetime'] = km_results['lpep_pickup_datetime'].apply(secs2datestr)
km_results['Lpep_dropoff_datetime'] = km_results['Lpep_dropoff_datetime'].apply(secs2datestr)

# Write the resultant points to a csv.
km_results.to_csv('../web/csv/km_results.csv', index=False)

