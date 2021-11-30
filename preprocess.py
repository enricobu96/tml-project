import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.metrics import ClassificationMetric

pd.set_option('display.max_columns', None)
_night = ['22:', '23:', '00:', '01:', '02:', '03:', '04:', '05:', '06:', '07:']

data = pd.read_csv('data/police_project.csv')
data = data.drop(['stop_date','county_name', 'driver_age_raw', 'violation_raw', 'stop_outcome', 'drugs_related_stop', 'search_type'], axis=1)
data = data.dropna()

data['stop_time'] = data['stop_time'].apply(lambda x: 0 if any(n in x for n in _night) else 1)

data['driver_gender'] = data['driver_gender'].astype('category').cat.codes
data['driver_race'] = data['driver_race'].astype('category').cat.codes
data['violation'] = data['violation'].astype('category').cat.codes
data['search_conducted'] = data['search_conducted'].astype('category').cat.codes
data['is_arrested'] = data['is_arrested'].astype('category').cat.codes
data['stop_duration'] = data['stop_duration'].astype('category').cat.codes
# #print(data.shape)

X = data[data.columns.difference(['is_arrested'])]
y = data['is_arrested']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression().fit(X_train, y_train)

y_pred = clf.predict(X_test)

priv_groups = [{'driver_race': 4}]
unpriv_groups = [{'driver_race': [0,1,2,3]}]

entire_test = pd.concat([X_test, y_test], axis=1)
entire_test_pred = pd.concat([X_test, y_pred], axis=1)

class_metric = ClassificationMetric(entire_test, y_pred, unprivileged_groups=unpriv_groups, privileged_groups=priv_groups)
print(class_metric.false_omission_rate())