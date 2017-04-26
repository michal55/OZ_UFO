# http://pandas.pydata.org/pandas-docs/stable/10min.html
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('scrubbed.csv', low_memory = False)
df['datetime'] = df['datetime'].apply(lambda x: x.replace('24:00', '23:59'))
df['datetime'] = pd.to_datetime(df['datetime'], format = '%m/%d/%Y %H:%M')


def time_of_day(datetime):
    if datetime.hour < 5 or datetime.hour > 21:
        return 'night'
    elif datetime.hour < 11:
        return 'morning'
    elif datetime.hour < 14:
        return 'noon'
    elif datetime.hour < 18:
        return 'afternoon'
    elif datetime.hour < 22:
        return 'evening'

    assert False


df['time_of_day'] = df['datetime'].map(time_of_day, na_action = 'ignore')

print(df[['datetime', 'time_of_day']])

df['time_of_day'].groupby(df.time_of_day).count().plot(kind = 'bar')
plt.show()
