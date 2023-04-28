#Python file that calls that loads training and testing datasets and calls LSSCV.py from lssvm to:
# 1. time to fit model to training data
# 2. accuracy of prediction on testing data

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler

from lssvm import LSSVC, LSSVC_GPU
from utils.encoding import dummie2multilabel
import time


# X_tr_norm = np.load('../X_train_values.npy')
# X_ts_norm = np.load('../X_test_values.npy')
# y_train = np.load('../y_train_values.npy')
# y_test = np.load('../y_test_values.npy')



X_tr_norm = np.load('../../data_final/X_train_moon_12500_0.3.npy')
X_ts_norm = np.load('../../data_final/X_test_moon_12500_0.3.npy')
y_train = np.load('../../data_final/y_train_moon_12500_0.3.npy')
y_test = np.load('../../data_final/y_test_moon_12500_0.3.npy')


# # Get information about input and outputs
# print(f"X_train.shape: {X_train.shape}")
# print(f"X_test.shape:  {X_test.shape}")
# print(f"y_train.shape: {y_train.shape}")
# print(f"y_test.shape:  {y_test.shape}")
# print(f"np.unique(y_train): {np.unique(y_train)}")
# print(f"np.unique(y_test):  {np.unique(y_test)}")


# # Use the classifier with different kernels

# print('Gaussian kernel:')
# lssvc = LSSVC(gamma=1, kernel='rbf', sigma=1) # Class instantiation
# lssvc.fit(X_tr_norm, y_train) # Fitting the model
# y_pred = lssvc.predict(X_ts_norm) # Making predictions with the trained model
# acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred)) # Calculate Accuracy
# print('acc_test using Gaussian Kernel = ', acc*100,'\n')

print('Polynomial kernel:')
lssvc = LSSVC(gamma=1, kernel='poly', d=3)
tic=time.perf_counter()
lssvc.fit(X_tr_norm, y_train)
toc=time.perf_counter()
print("time = ", toc-tic)  
y_pred = lssvc.predict(X_ts_norm)
acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred))
print('acc_test using Polynomial Kernel = ', acc*100, '\n')

# print('Linear kernel:')
# lssvc = LSSVC(gamma=1, kernel='linear')
# lssvc.fit(X_tr_norm, y_train)
# y_pred = lssvc.predict(X_ts_norm)
# acc = accuracy_score(dummie2multilabel(y_test), dummie2multilabel(y_pred))
# print('acc_test using Linear Kernel = ', acc*100, '\n')

# lssvc.dump('model')
# loaded_model = LSSVC.load('model')

# # Showing the same results
# print('acc_test = ', accuracy_score(
#         dummie2multilabel(y_test), 
#         dummie2multilabel(lssvc.predict(X_ts_norm))
#     )
# )
# print('acc_test = ', accuracy_score(
#         dummie2multilabel(y_test), 
#         dummie2multilabel(loaded_model.predict(X_ts_norm))
#     )
# )

