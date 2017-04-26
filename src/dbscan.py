import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap

from preprocessing import preprocess_ufo_data

df = None
if len(sys.argv) == 2 and sys.argv[1] == 'load':
    if not os.path.isfile('preprocessed.csv'):
        print('preprocessed.csv is not present in working directory, consider running "python3 preprocessing.py save" first')
        os.exit(1)

    df = pd.read_csv('preprocessed.csv', low_memory = False)
else:
    df = pd.read_csv('scrubbed.csv', low_memory = False)

    print('Preprocessing data, consider running "python3 preprocessing.py save" once and then start this script with "load" argument')
    start = time.time()
    # Apply preprocessing, in preprocessing.py file
    df = preprocess_ufo_data(df)
    print('Preprocessing finished in', time.time() - start, 'seconds')


# Construct map (background)
# Default 'map' projection
# map = Basemap(projection='robin', lat_0=0, lon_0=-0, resolution='l', area_thresh=1000.0)
# Globe - American side
map = Basemap(projection='nsper', lat_0=40, lon_0=-105, resolution='l', area_thresh=1000.0)
# Globe - EU side
# map = Basemap(projection='nsper', lat_0=40, lon_0=-0, resolution='l', area_thresh=1000.0)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='grey')
map.drawmapboundary()
 
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

# Plot raw values from csv
#x, y = map(df[['longitude ']].values, df[['latitude']].values)
#map.plot(x, y, 'bo')

#plt.show()


# Reduce out dataset to finish fast
df = df.drop(df.index[range(0, 30000)])
column_name = ""

# print(len(df.query("(shape == column_name)").values))
# os.exit[0]
# Reduce dataset to features used for DBSCAN
# df = df[['latitude', 'longitude', 'timestamp']]
df = pd.concat([df[['latitude', 'longitude', 'timestamp']], pd.get_dummies(df['shape'])], axis=1)
print(df.columns.values)
column_len = len(df.columns.values)

# Principal component analysis - dimension reduction - not needed for <100 columns
# pca = PCA(n_components=5)
# pca.fit_transform(df)


# Scaling magic
scaler = StandardScaler().fit(df)
df = scaler.transform(df)


# DBSCAN
db = DBSCAN(eps=0.3, min_samples=20, algorithm='auto').fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print('Core samples: ', core_samples_mask)
# Reverse scaling so we can plot real coordinates
df = scaler.inverse_transform(df)



# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

print(unique_labels)

for k, col in zip(unique_labels, colors):
    if k == -1:
        continue
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = df[class_member_mask & core_samples_mask]
    
    

    # This line transforms world coordinates to map coordinates
    x, y = map(xy[:, 1], xy[:, 0])

    # map.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    xy = df[class_member_mask & ~core_samples_mask]

    # This line transforms world coordinates to map coordinates
    x, y = map(xy[:, 1], xy[:, 0])

    # map.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=4)


    cluster = df[class_member_mask]

    # print(cluster)
    # print(len(df.query("(shape == column_name)").values))
    cluster=cluster[:, range(3,column_len)]
    print(cluster.sum(axis=0))
    plt.plot(cluster.sum(axis=0), 'ro')
    plt.show()

    

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

