# # n_samples = 100
# # n_features = 200
#Python file to make data for classification. Used to measure time performance and accuracy of diiferent SVM 
#models

# # centers = 2

# # X, y = make_blobs(n_samples=n_samples, n_features=n_features,centers=centers,random_state=0, cluster_std=0.50)
# # X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# # dump_svmlight_file(X=X_train,y=y_train,f=f'Train_{n_samples}samples_{n_features}features_{centers}centers.dat',zero_based=False)
# # dump_svmlight_file(X=X_test,y=y_test,f=f'Test_{n_samples}samples_{n_features}features_{centers}centers.dat',zero_based=False)


# ######################################################################################
import numpy as np
import sklearn
import sklearn.datasets._samples_generator
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split
# from sklearn.datasets._samples_generator import make_circles
# from sklearn.datasets._samples_generator import make_friedman1
from sklearn.datasets._samples_generator import make_moons
# from sklearn.datasets._samples_generator import make_classification

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
n_samples = 12500
n_features = 200
noise = 0.3
X, y = make_moons(n_samples=n_samples,noise=noise,random_state=2)
# X, y = make_classification(n_samples=n_samples,n_features=n_features,n_classes=2,n_clusters_per_class=1,random_state=2)
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
scaler.fit(X_train)
X_tr_norm = scaler.fit_transform(X_train)
X_te_norm = scaler.transform(X_test)


# dump_svmlight_file(X=X_train,y=y_train,f='Train_moon.dat',zero_based=False)
# dump_svmlight_file(X=X_test,y=y_test,f='Test_moon.dat',zero_based=False)

np.save(f'data_final/X_train_moon_{n_samples}_{noise}',X_tr_norm)
np.save(f'data_final/X_test_moon_{n_samples}_{noise}',X_te_norm)
np.save(f'data_final/y_train_moon_{n_samples}_{noise}',y_train)
np.save(f'data_final/y_test_moon_{n_samples}_{noise}',y_test)


################################################################
# import numpy as np

# A = np.array([
#     [1,2,3,14],
#     [4,5,6,15],
#     [7,8,9,16],
#     [10,11,12,17]
# ])
# y_values = [1,2,3]
# B = np.array([0]+[1]*len(y_values))



# # np.savetxt('A_array.txt',A)


# f = open('A_array.txt','a')
# np.savetxt(f,[len(A)],fmt='%1.3f')
# np.savetxt(f,A)



