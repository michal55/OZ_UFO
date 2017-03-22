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

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 500)

plt.rcParams.update({'font.size': 14})

df = pd.read_csv('scrubbed.csv',  dtype='unicode')
print(df.columns)
# df.query('state==tx')

# print(df['shape'].groupby('shape').agg(['count']))

# print(df.groupby('state').value_counts())
# print(df['shape'].value_counts())
# df = 
plt.figure()
# df = df['shape'].value_counts()
# print(df)
# df[df > 50].plot.bar()

# df = df['duration (seconds)']


dc = df['duration (seconds)'].value_counts().to_dict()
# df = df.query("(duration (seconds) < 60)")
# .value_counts()

d = dict((k, v) for k, v in dc.items() if k <= '60')
print(d)


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