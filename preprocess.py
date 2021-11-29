import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

pd.set_option('display.max_columns', None)

data = pd.read_csv('data/police_project.csv')
data = data.drop(['stop_date','county_name', 'driver_age_raw', 'violation_raw', 'stop_outcome', 'drugs_related_stop'], axis=1)
#data['stop_time'] = data['stop_time'].apply(lambda x: 0 if (datetime.strptime(x, '%H:%M') > datetime(year=1, month=1, day=1, hour=22, minute=00) and datetime.strptime(x, '%H:%M') <= datetime(year=3000, day=31, month=12, hour=7, minute=00)) else 1)
#print(data['stop_time'].unique())

data['driver_gender'] = data['driver_gender'].apply(lambda x: 0 if x == 'M' else 1)
data['driver_age'] = data['driver_age'].apply(lambda x: print(type(x)))
