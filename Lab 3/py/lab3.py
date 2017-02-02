import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import time
from matplotlib.dates import date2num
import math

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

num_items = 5000

# Method for converting the datetime to seconds since epoch.
def datestr2secs(s):
    return (datetime.strptime(s, '%Y-%m-%d %H:%M:%S')-datetime(1970,1,1)).total_seconds()

def secs2datestr(s):
    return (datetime(1970, 1, 1) + timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S')

# Read in the taxi data.
taxi_sheet = pd.read_csv('../web/csv/km_results.csv')
attributes = np.array(taxi_sheet.columns.values.tolist())

# Convert the date time strings to seconds since epoch.
taxi_sheet['lpep_pickup_datetime'] = taxi_sheet['lpep_pickup_datetime'].apply(datestr2secs)
taxi_sheet['Lpep_dropoff_datetime'] = taxi_sheet['Lpep_dropoff_datetime'].apply(datestr2secs)

# Perform PCA analysis.
numeric_data = taxi_sheet._get_numeric_data()
pca = PCA()
pca.fit(numeric_data)
proj_data = pca.transform(numeric_data)
proj_data_df = pd.DataFrame(proj_data[:,:2], columns=range(1,3))
proj_data_df.to_csv('../web/csv/proj_data.csv', index=False)

# Calculate MDS using the euclidean distance.
mds = MDS(n_components=2, dissimilarity="euclidean")
mds_data = mds.fit_transform(numeric_data)
mds_data_df = pd.DataFrame(mds_data[:,:2], columns=range(1,3))
mds_data_df.to_csv('../web/csv/mds_data.csv', index=False)

# Obtain the covariance matrix.
cov = np.cov(numeric_data.T)

# Grab the eigenvalues and eigenvectors.
eigenvals, eigenvecs = np.linalg.eig(cov)
eigenvecs_df = pd.DataFrame(eigenvecs[:,:2], columns=range(1, 3))
eigenvecs_df['attribute'] = attributes
eigenvecs_df.to_csv('../web/csv/eigenvecs.csv', index=False)

# Obtain the scree plot values.
eigenvals_sorter = np.argsort(eigenvals)
eigenvals_sorted = eigenvals[eigenvals_sorter][::-1]
eigenvals_percentage = []
for i in range(0, len(eigenvals_sorted)):
    eigenvals_percentage.append(eigenvals_sorted[i] / np.sum(eigenvals_sorted) * 100)
eigenvals_percentage = np.array(eigenvals_percentage)
eigenvals_percentage_df = pd.DataFrame(eigenvals_percentage, columns=['percentage'])
eigenvals_percentage_df['component'] = pd.DataFrame([str(x) for x in range(1, 11)])
eigenvals_percentage_df.to_csv('../web/csv/scree.csv', index=False)

# Convert the taxi_sheet times back to strings.
taxi_sheet['lpep_pickup_datetime'] = taxi_sheet['lpep_pickup_datetime'].apply(secs2datestr)
taxi_sheet['Lpep_dropoff_datetime'] = taxi_sheet['Lpep_dropoff_datetime'].apply(secs2datestr)

# Calculate the sum of squared loadings.
squared_loadings = []
for i in range(0, len(eigenvecs)):
    squared_loadings.append(np.sum(np.square(eigenvecs[i][:10])) ** (10 ** 14))

# Order features by contribution to most significant PCA vectors.
squared_loadings = np.array(squared_loadings)
sorter = np.argsort(squared_loadings)
attributes_sorted = attributes[sorter][::-1]
squared_loadings_sorted = squared_loadings[sorter][::-1]

# Write the top five attributes.
top_five = taxi_sheet[attributes_sorted[:5]]
print(attributes_sorted)
print(squared_loadings_sorted)
top_five.to_csv('../web/csv/results_top_five.csv', index=False)

# Obtain the correlation matrix.
cor = np.corrcoef(numeric_data.T)
cor_df = pd.DataFrame(cor, columns=attributes)
cor_df.to_csv('../web/csv/cor.csv', index=False)

# Calculate MDS using the 1 - |correlation| distance.
dis_df = cor_df.applymap(lambda x: 1 -abs(x))
mds2 = MDS(n_components=2, dissimilarity="precomputed")
mds2_data = mds2.fit_transform(dis_df)
mds2_data_df = pd.DataFrame(mds2_data[:,:2], columns=range(1,3))
mds2_data_df['attribute'] = attributes
mds2_data_df.to_csv('../web/csv/mds2_data.csv', index=False)