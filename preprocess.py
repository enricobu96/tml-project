import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from aif360.metrics import ClassificationMetric
from aif360.datasets import StandardDataset, dataset
from copy import copy
pd.set_option('display.max_columns', None)
_night = ['22:', '23:', '00:', '01:', '02:', '03:', '04:', '05:', '06:', '07:']

# data preprocessing
data = pd.read_csv('data/police_project.csv')
data = data.drop(['stop_date','county_name', 'driver_age_raw', 'violation_raw', 'stop_outcome', 'drugs_related_stop', 'search_type'], axis=1)
data = data.dropna()
data['stop_time'] = data['stop_time'].apply(lambda x: 0 if any(n in x for n in _night) else 1)
data['driver_gender'] = data['driver_gender'].astype('category').cat.codes
data['driver_race'] = data['driver_race'].astype('category').cat.codes
data['driver_race'] = data['driver_race'].apply(lambda x: 0 if x==3 else 1)
data['violation'] = data['violation'].astype('category').cat.codes
data['search_conducted'] = data['search_conducted'].astype('category').cat.codes
data['is_arrested'] = data['is_arrested'].astype('category').cat.codes # 0 = False, 1 = True
data['stop_duration'] = data['stop_duration'].astype('category').cat.codes

# # add bias
# def perform_bias(x):
#     r_ch_nw = np.random.choice([0,1], p=[.8,.2])
#     r_ch_w = np.random.choice([0,1], p=[.2,.8])
#     if x == 0 and r_ch_nw == 0:
#         return True
#     elif x == 0 and r_ch_nw == 1:
#         return False
#     elif x == 1 and r_ch_w == 0:
#         return True
#     return False
        
# data['is_arrested'] = data['driver_race'].apply(lambda x: perform_bias(x))

# datasets
X = data[data.columns.difference(['is_arrested'])]
y = data['is_arrested']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ai360
priv_groups = [{'driver_race': 1}]
unpriv_groups = [{'driver_race': 0}]
dataset = pd.concat([X_test, y_test], axis=1)
dataset = StandardDataset(dataset, label_name='is_arrested', favorable_classes=[0], protected_attribute_names=['driver_race'], privileged_classes=[[1]])
classified_dataset = dataset.copy()
classified_dataset.labels = y_pred

# metrics
class_metric = ClassificationMetric(dataset, classified_dataset, unprivileged_groups=unpriv_groups, privileged_groups=priv_groups)
print(class_metric.false_omission_rate())