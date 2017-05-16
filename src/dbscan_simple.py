import random
import os
import pickle
import time
import sys
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

import jqmcvi.base as jqmcvi
from preprocessing import preprocess_ufo_data

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


def DBScan(df, eps, min_samples):
    file_name = 'dbscan_' + str(eps) + '_' + str(min_samples)

    if len(sys.argv) >= 3 and sys.argv[2] == 'skip' and not os.path.isfile(file_name):
        return 0.

    # DBSCAN
    db = None
    if os.path.isfile(file_name):
        db = pickle.load(open(file_name, 'rb'))
    else:
        db = DBSCAN(eps = eps, min_samples = min_samples, algorithm='auto').fit(df)
        pickle.dump(db, open(file_name, 'wb'))

    labels = db.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print('Clusters for', eps, min_samples, n_clusters)

    if n_clusters == 0:
        return 0.

    clusters = []
    centers = []
    in_cluster_sim = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue

        class_member_mask = (labels == k)

        cluster = df[class_member_mask]
        clusters.append(cluster)

        center = np.average(cluster, axis = 0)
        centers.append(center)

        in_cluster_sim.append(cluster_sim(center, cluster))

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'db':
            return jqmcvi.davisbouldin(clusters, centers)
        elif sys.argv[1] == 'custom':
            cluster_similarities = similarity(centers, "else")

            sum_cross_cluster = sum(cluster_similarities)
            avg_cross_sum = sum_cross_cluster / float(len(centers))

            print("Average cross-cluster similarity: ", avg_cross_sum)

            intra_sum = sum(in_cluster_sim)
            avg_intra_sum = intra_sum/float(len(in_cluster_sim))

            print("Average intra-cluster similarity: ", avg_intra_sum)
            print("Number of clusters: ", len(centers))
            print("\nIntra / Cross: ", avg_intra_sum / avg_cross_sum)
            return min(4.0, avg_intra_sum / avg_cross_sum)

    # Evaluate clustering
    return random.random()



df = pd.read_csv('scrubbed.csv', low_memory = False)

# Apply preprocessing, in preprocessing.py file
df = preprocess_ufo_data(df)

# Almost all columns
df = pd.concat([df[['latitude', 'longitude', 'timestamp', 'season_x', 'season_y', 
 'day_of_year_x', 'day_of_year_y', 'hour_of_day', 'hour_of_day_x',
 'hour_of_day_y' ]], pd.get_dummies(df['shape']), pd.get_dummies(df['season']), pd.get_dummies(df['time_of_day'])], axis=1)

# Scaling magic
scaler = StandardScaler().fit(df)
df = scaler.transform(df)

MIN_SAMPLES_START = 10
MIN_SAMPLES_END = 100
MIN_SAMPLES_COUNT = 10

EPS_START = 0.3
EPS_END = 3.0
EPS_COUNT = 10

x_data = np.linspace(MIN_SAMPLES_START, MIN_SAMPLES_END, MIN_SAMPLES_COUNT)
y_data = np.linspace(EPS_START, EPS_END, EPS_COUNT)
z_data = [[] for _ in range(len(x_data))]

start = time.time()


def solve_for(idx):
    global z_data
    for eps in y_data:
        z_data[idx].append(DBScan(df, eps, x_data[idx]))
    print('Solved for', idx, 'elapsed', time.time() - start, 'seconds')

'''with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as e:
    arr = [e.submit(solve_for, i) for i in range(0, len(x_data))]
    concurrent.futures.wait(arr)
'''

#[solve_for(i) for i in range(0, len(x_data))]

z_data = pickle.load(open('custom_z', 'rb'))
#pickle.dump(z_data, open('custom_z', 'wb'))

print('Plotting...')

z_data = np.array(z_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data, y_data = np.meshgrid(x_data, y_data)

x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = z_data.flatten()
#ax.bar3d(x_data, y_data, np.zeros(len(z_data)),
#        (MIN_SAMPLES_END - MIN_SAMPLES_START) / MIN_SAMPLES_COUNT - 1.,
#        (EPS_END - EPS_START) / EPS_COUNT - 0.07, z_data )
ax.plot_surface(x_data, y_data, z_data, rstride = 1, cstride = 1)

ax.set_xlabel('min samples')
ax.set_ylabel('eps')
ax.set_zlabel('value')

print('Finished in', time.time() - start, 'seconds')
plt.show()
