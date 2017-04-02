import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap

df = pd.read_csv('scrubbed.csv', low_memory = False)

# Reduce out dataset to finish fast
df = df.drop(df.index[range(0, 60000)])

# Parse datetime
df['datetime'] = df['datetime'].apply(lambda x: x.replace('24:00', '23:59'))
df['datetime'] = pd.to_datetime(df['datetime'], format = '%m/%d/%Y %H:%M')


# Set all non parsable values for coordinates to 0 - FIXME
def hack_floats(num):
    try:
        return float(num)
    except:
        return 0.0

df['latitude'] = df['latitude'].map(hack_floats, na_action = 'ignore')
df['longitude '] = df['longitude '].map(hack_floats, na_action = 'ignore')

# Create timestamp feature from datetime
def to_timestamp(datetime):
    return time.mktime(datetime.timetuple())

df['timestamp'] = df['datetime'].map(to_timestamp, na_action = 'ignore')

# Construct map (background)
map = Basemap(projection='robin', lat_0=0, lon_0=-100, resolution='l', area_thresh=1000.0)
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral')
map.drawmapboundary()
 
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

# Plot raw values from csv
#x, y = map(df[['longitude ']].values, df[['latitude']].values)
#map.plot(x, y, 'bo')

#plt.show()




# Reduce dataset to features used for DBSCAN
df = df[['latitude', 'longitude ', 'timestamp']]

# Scaling magic
scaler = StandardScaler().fit(df)
df = scaler.transform(df)

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(df)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# Reverse scaling so we can plot real coordinates
df = scaler.inverse_transform(df)



# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
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

    map.plot(x, y, 'o', markerfacecolor=col, markeredgecolor='k', markersize=2)


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

