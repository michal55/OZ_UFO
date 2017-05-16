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
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.spatial import distance
from mpl_toolkits.basemap import Basemap

# import jqmcvi.base as jqmcvi

from preprocessing import preprocess_ufo_data

pd.options.display.width = 180
# np.set_printoptions(precision=3)
# np.set_printoptions(suppress=True)

def similarity(array, output):
    sim = []
    for X in range(len(array)):
        sim.append([])
        for Y in range(len(array)):
            sim[X].append((distance.cosine(array[X], array[Y])+1)**2)
            # print(sim[X][Y])
    if output == "arr":
        return sim
    else:
        summed = []
        for X in range(len(array)):
            summed.append(sum(sim[X]))
        return summed           


def cluster_sim(centroid, cluster):
    sim = []
    for element in cluster:
        sim.append((distance.cosine(centroid, element)+1)**2)
    return sum(sim)



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

# print(df.columns.values)
# print(df)
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

# df = df.drop(df.index[range(0, 40000)])
column_name = ""

# print(len(df.query("(shape == column_name)").values))
# os.exit[0]
# Reduce dataset to features used for DBSCAN
# df = df[['latitude', 'longitude', 'timestamp']]

# Ignored columns for now
# , 'season_num''day_of_year',

# Default settings
# df = pd.concat([df[['latitude', 'longitude', 'timestamp']], pd.get_dummies(df['shape'])], axis=1)

# Binary seasons, time of day....
# df = pd.concat([df[['latitude', 'longitude', 'timestamp']], 
#     pd.get_dummies(df['shape']), pd.get_dummies(df['season']), 
#     pd.get_dummies(df['hour_of_day']), pd.get_dummies(df['time_of_day'])], axis=1)

# season a day of year korelacia

# Numeric attributes
df = pd.concat([df[['latitude', 'longitude', 'timestamp', 'season_x', 'season_y', 
 'day_of_year_x', 'day_of_year_y', 'hour_of_day_x',
 'hour_of_day_y' ]], pd.get_dummies(df['shape'])], axis=1)



column_names = df.columns.values
print(df.columns.values)
column_len = len(df.columns.values)

# Principal component analysis - dimension reduction - not needed for <100 columns
# pca = PCA(n_components=5)
# pca.fit_transform(df)

# Uncomment to use nicely clusterable sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#df, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.2, random_state=0)


# Scaling magic
scaler = StandardScaler().fit(df)
df = scaler.transform(df)


# DBSCAN
db = DBSCAN(eps=0.1, min_samples=20, algorithm='auto').fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# Evaluation

# With visualization TODO http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, labels))

# Setup
#git clone https://github.com/jqmviegas/jqm_cvi
#cd jqm_cvi
#sudo pip3 install cython
#sudo python3 setup.py install

# Evaluation code
# print(jqmcvi.dunn_fast(df, labels))



# Reverse scaling so we can plot real coordinates
df = scaler.inverse_transform(df)

# Dictionary of variances for each cluster
variances = {}
vectors = []

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

print(unique_labels)

in_cluster_sim = []

for k, col in zip(unique_labels, colors):
    if k == -1:
        continue
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = df[class_member_mask & core_samples_mask]
    
    

    # This line transforms world coordinates to map coordinates
    x, y = map(xy[:, 1], xy[:, 0])

    map.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    xy = df[class_member_mask & ~core_samples_mask]

    # This line transforms world coordinates to map coordinates
    x, y = map(xy[:, 1], xy[:, 0])

    map.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=4)


    cluster = df[class_member_mask]

    print(cluster)
    # print(len(df.query("(shape == column_name)").values))

    # Only sum binary columns
    # cluster=cluster[:, range(10,column_len)]

    # Sum of binary columns
    # cluster_values = cluster.sum(axis=0).astype(int)
    
    # Avg of numeric columns - Centroid
    cluster_values = np.average(cluster,axis=0)

    # Downscale timestamp parameter
    cluster_values[2] /= 1000000000
    
    # 
    in_cluster_sim.append(cluster_sim(cluster_values, cluster))
    vectors.append(cluster_values)
    
    # plt.plot(cluster.sum(axis=0), 'ro')
    # plt.show()

print(column_names)

cluster_similarities = similarity(vectors, "else")

# print("Sum of similarities to other clusters for each cluster")

# for num in cluster_similarities:
#     print(num)

sum_cross_cluster = sum(cluster_similarities)
avg_cross_sum = sum_cross_cluster/float(len(vectors))

print("Average cross-cluster similarity: ", avg_cross_sum)

intra_sum = sum(in_cluster_sim)
avg_intra_sum = intra_sum/float(len(in_cluster_sim))

print("Average intra-cluster similarity: ", avg_intra_sum)
print("Number of clusters: ", len(vectors))
print("\nIntra / Cross: ", avg_intra_sum/avg_cross_sum)

plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

# porovnat vektory kazdeho pozorovania s centroidom - priemerny vektor z results.txt
# porovnat centroidy medzi sebou