import random
import time
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from preprocessing import preprocess_ufo_data


def DBScan(df, eps, min_samples):
    # Scaling magic
    scaler = StandardScaler().fit(df)
    df = scaler.transform(df)

    # DBSCAN
    db = DBSCAN(eps = eps, min_samples = min_samples, algorithm='auto').fit(df)

    # Reverse scaling if we need it for calculations
    #df = scaler.inverse_transform(df)

    # Evaluate clustering
    return random.random()



df = pd.read_csv('scrubbed.csv', low_memory = False)

# Apply preprocessing, in preprocessing.py file
df = preprocess_ufo_data(df)

# Almost all columns
df = pd.concat([df[['latitude', 'longitude', 'timestamp', 'season_x', 'season_y', 
 'day_of_year_x', 'day_of_year_y', 'hour_of_day', 'hour_of_day_x',
 'hour_of_day_y' ]], pd.get_dummies(df['shape']), pd.get_dummies(df['season']), pd.get_dummies(df['time_of_day'])], axis=1)

MIN_SAMPLES_START = 1
MIN_SAMPLES_END = 100
MIN_SAMPLES_COUNT = 4

EPS_START = 0.1
EPS_END = 1
EPS_COUNT = 4

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

[solve_for(i) for i in range(0, len(x_data))]

print('Plotting...')

z_data = np.array(z_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data, y_data = np.meshgrid(x_data, y_data)

x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = z_data.flatten()
ax.bar3d(x_data, y_data, np.zeros(len(z_data)),
        (MIN_SAMPLES_END - MIN_SAMPLES_START) / MIN_SAMPLES_COUNT,
        (EPS_END - EPS_START) / EPS_COUNT, z_data )

ax.set_xlabel('min samples')
ax.set_ylabel('eps')
ax.set_zlabel('value')

print('Finished in', time.time() - start, 'seconds')
plt.show()
