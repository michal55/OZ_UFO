import time
import sys
from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def season(dt):
    # http://stackoverflow.com/a/28688724/6022799
    Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

    dt = dt.date().replace(year=Y)
    return next(season for season, (start, end) in seasons if start <= dt <= end)

# Set all non parsable values for coordinates to 0
def fix_floats(num):
    try:
        return float(num)
    except:
        return 0.0

def to_timestamp(datetime):
    return time.mktime(datetime.timetuple())

def preprocess_ufo_data(df):
    # Parse datetime and date posted
    df['datetime'] = df['datetime'].apply(lambda x: x.replace('24:00', '23:59'))
    df['datetime'] = pd.to_datetime(df['datetime'], format = '%m/%d/%Y %H:%M')

    df['date posted'] = pd.to_datetime(df['date posted'], format = '%m/%d/%Y')

    # Create time_of_day and season attributes
    df['time_of_day'] = df['datetime'].map(time_of_day, na_action = 'ignore')
    df['season'] = df['datetime'].map(season, na_action = 'ignore')

    # Create timestamp attribute from datetime
    df['timestamp'] = df['datetime'].map(to_timestamp, na_action = 'ignore')

    # Remove empty shapes
    df = df.query("(shape == shape)")

    # Fix floats and remove trailing space from longitude column name
    df['latitude'] = df['latitude'].map(fix_floats, na_action = 'ignore')
    df['longitude'] = df['longitude '].map(fix_floats, na_action = 'ignore')

    # Remove sightings remorted more than 90 days after the event
    df = df.drop(df[(df['date posted'] - df['datetime']).dt.days > 90].index)

    return df


def main():
    df = pd.read_csv('scrubbed.csv', low_memory = False)
    df = preprocess_ufo_data(df)

    if len(sys.argv) == 2:
        if sys.argv[1] == 'save':
            df.to_csv('preprocessed.csv')
        elif sys.argv[1] == 'plot':
            #df['time_of_day'].groupby(df.time_of_day).count().plot(kind = 'bar')
            df['season'].groupby(df.season).count().plot(kind = 'bar')
            plt.show()

if __name__ == "__main__":
    main()

