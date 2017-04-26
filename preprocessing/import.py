import csv
# with open('complete.csv') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=',')
# 	for row in spamreader:
# 		print('\n')
# 		for word in row:
# 			print('|' + word, end="")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)

plt.rcParams.update({'font.size': 14})

df = pd.read_csv('/home/michal/workspace/OZ/scrubbed.csv',  dtype='unicode')
print(df.columns)
# df.query('state==tx')

# print(df['shape'].groupby('shape').agg(['count']))

# print(df.groupby('state').value_counts())
# print(df['shape'].value_counts())

count = 0
def shape_from_comment( row ):
	global count
	comment = row['comments']
	shapes = ['cylinder', 'light', 'circle', 'sphere', 'disk', 'fireball', 'unknown', 'oval',
	 'other', 'cigar', 'rectangle', 'chevron', 'triangle', 'formation', 'delta',
	 'changing', 'egg', 'diamond', 'flash', 'teardrop', 'cone', 'cross', 'pyramid',
	 'round', 'crescent', 'flare', 'hexagon', 'dome', 'changed']
	if isinstance(comment, str):
		result = (' ').join(set(shapes).intersection(set(comment)))
		for shape in shapes:
			if shape in comment:
				result.join(shape)
		if len(result) > 1:
			return result
		else:
			# print(comment)
			return np.nan
	else:
		count += 1
		print(comment)
		return "None"

def first_word( row ):
	if isinstance(row['comments'], str):
		return (row['comments'].partition(' ')[0])
	else:
		return "None"

plt.figure()
# shapes = df['shape'].unique()
# print(shapes)	

# comments = df['comments']
# print(comments.size)
# print(len(df))

# emtpy_shapes = df.query("(shape != shape)")
# print(emtpy_shapes.size)

# emtpy_shapes['shape'] = emtpy_shapes.apply(shape_from_comment, axis=1)
# print(emtpy_shapes)
# print(emtpy_shapes)

# print(df.query("(shape != shape)").size)

df = df.query("(shape == shape)")

# df = pd.get_dummies(df['shape'])
pca = PCA(n_components = 5)
P = pca.fit(pd.get_dummies(df['shape']))

# print(df)

# print(df['longitude', 'latitude', 'timestamp'])

print(pd.concat([df[['longitude', 'latitude']], P], axis=1))

# print(pd.get_dummies(df['shape']))
# print(count)

# print(df[[(not isinstance(x, str)) for x in df['comments']]])

# print(df.select_dtypes(include=['float64']))

# type(df['shape'])

# first_word(emtpy_shapes['comments'].iloc[1])
# shape_from_comment(df['comments'][0])
# emtpy_shapes['shape'] = emtpy_shapes['']



# df[df > 50].plot.bar()
# df = df['duration (seconds)']


# dc = df['duration (seconds)'].value_counts().to_dict()
# df = df.query("(duration (seconds) < 60)")
# .value_counts()

# d = dict((k, v) for k, v in dc.items() if k <= '60')
# print(d)


# df = df['duration (seconds)'].value_counts().hist(bins=list(range(0,60,1))).figure
# df = df['duration (seconds)'].value_counts().hist(bins=list(range(100,2500,100))).figure
# print(df)
# df.hist()
# print(df)


# df.plot(x='A', y='B')
# plt.plot()
# print(df.select(df.columns.filter())
# n = np.nan
# print(np.nan)
# print(df.fillna('missing').query("(longitude == 'missing')"))

# print(df['latitude'])
# print(df.query("(latitude == None)"))

# df['DateSize'] = df['date posted'].str.len()
# print(df.query("(DateSize > 9) & (city == 'morland to hill city')"))
# print(df)


# evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Count')
# plt.show()

# print(df[df['latitude'] == 'NaN'])