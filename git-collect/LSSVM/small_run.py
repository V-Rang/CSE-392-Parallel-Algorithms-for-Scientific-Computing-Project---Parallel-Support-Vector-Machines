#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

from lssvm import LSSVC, LSSVC_GPU
from utils.encoding import dummie2multilabel


# In[ ]:





# In[ ]:


# Preprocessing

# Import digits recognition dataset (from sklearn)
X, y = load_digits(return_X_y=True)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2020)
X_train = np.load('data_final_multi_feature/X_train_moon_2000_0.3.npy')
X_test = np.load('data_final_multi_feature/X_test_moon_2000_0.3.npy')
y_test= np.load('data_final_multi_feature/y_test_moon_2000_0.3.npy')
y_train= np.load('data_final_multi_feature/y_train_moon_2000_0.3.npy')
# Scaling features (from sklearn)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_tr_norm = scaler.transform(X_train)
X_ts_norm = scaler.transform(X_test)

# Get information about input and outputs
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape:  {X_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape:  {y_test.shape}")
print(f"np.unique(y_train): {np.unique(y_train)}")
print(f"np.unique(y_test):  {np.unique(y_test)}")


# In[ ]:


# Use the classifier with different kernels

"""print('Gaussian kernel:')
lssvc = LSSVC(gamma=1, kernel='rbf', sigma=.5) # Class instantiation
lssvc.fit(X_tr_norm, y_train) # Fitting the model
y_pred = lssvc.predict(X_ts_norm) # Making predictions with the trained model
acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred)) # Calculate Accuracy
print('acc_test = ', acc, '\n')"""

print('Polynomial kernel:')
lssvc = LSSVC(gamma=1, kernel='poly', d=2)
#method = 1 for CPU_SVD 2 for CPU_RSVD, 3 for Torch(GPU)_RSVD
method=3
lssvc.fit(X_tr_norm, y_train,method)
y_pred = lssvc.predict(X_ts_norm)
acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred))
print('acc_test = ', acc, '\n')

"""print('Linear kernel:')
lssvc = LSSVC(gamma=1, kernel='linear')
lssvc.fit(X_tr_norm, y_train)
y_pred = lssvc.predict(X_ts_norm)
acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred))
print('acc_test = ', acc, '\n')"""


# In[ ]:




