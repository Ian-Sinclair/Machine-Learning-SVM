# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:02:40 2021

@author: IanSi
"""
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784',version=1,return_X_y=True)


j= 10 
plt.title('The jth image is a {label}'.format(label=int(y[j])))
plt.imshow(X[j].reshape((28,28)),cmap='gray')
plt.show()

#Compare result with any without preprocessing.
X = preprocessing.scale(X)

X4 = X[y=='4',:]
X9 = X[y=='9',:]
y4 = 4*np.ones((len(X4),), dtype=int)
y9 = 9*np.ones((len(X9),), dtype=int)

X = np.concatenate((X4,X9),axis=0)
y = np.concatenate((y4,y9),axis=0)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.43,random_state=0)


'''
Training SVM using the holdout method
'''

#Partitions training data to sets used to train the classifier and test data for parameters.
X_fit,X_holdout,y_fit,y_holdout = train_test_split(X_train,y_train,test_size=0.3,random_state=0)


#Inhomogeneous linear SVM Classifier
min_error = 1;
C_ = 1

#Select C
for n in range(-14,15,1): #Training on range [0.00006, 16384 : *2].
    clf = svm.SVC(C=2**n,kernel='poly',degree = 1) #Builds linear classifier
    clf.fit(X_fit,y_fit) #Trains classifier using the partitioned training data
    Pe = 1-clf.score(X_holdout,y_holdout) #Tests classifier using the holdout set.
    if Pe < min_error: #Maintains the C_ value corresponding to the minimum error.
        min_error = Pe
        C_ = 2**n

print(C_)
#Retrain
clf = svm.SVC(C=C_,kernel='poly',degree = 1)
clf.fit(X_train,y_train)
#Error and support vectors
Pe = 1-clf.score(X_test,y_test)
num_SV = clf.support_vectors_



#Quadratic Kernel SVM Classifier
min_error = 1;
C_ = 1

#Select C
for n in range(-14,15,1): 
    clf = svm.SVC(C=2**n,kernel='poly',degree = 2) 
    clf.fit(X_fit,y_fit) 
    Pe = 1-clf.score(X_holdout,y_holdout) 
    if Pe < min_error: 
        min_error = Pe
        C_ = 2**n

print(C_)
#Retrain
clf = svm.SVC(C=C_,kernel='poly',degree = 2)
clf.fit(X_train,y_train)
#Error and support vectors
Pe = 1-clf.score(X_test,y_test)
num_SV = clf.support_vectors_




#Radial Basis Function
min_error = 1;
C_ = 1
gamma_ = 1

#Select C_ and gamma_
for k in range(-10,11,1):
    for n in range(-10,11,1): 
        clf = svm.SVC(C=2**n,kernel='rbf',gamma = 2**k) 
        clf.fit(X_fit,y_fit) 
        Pe = 1-clf.score(X_holdout,y_holdout) 
        if Pe < min_error: 
            min_error = Pe
            C_ = 2**n
            gamma_ = 2**k
            
print(C_)
print(gamma_)
#Retrain
clf = svm.SVC(C=C_,kernel='rbf',gamma = gamma_)
clf.fit(X_train,y_train)
#Error and support vectors
Pe = 1-clf.score(X_test,y_test)
num_SV = clf.support_vectors_