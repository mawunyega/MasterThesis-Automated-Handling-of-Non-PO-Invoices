# /* Computation of MSE for optimal K
#  *
#  * Filename: <MSE.py>
#  * Author  : <Lloyd .M. Dzokoto>
#  * Date    : <11.09.2018>
#  * Version : <1>
#  *
#  */


import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt


#creating odd list of K
mylist = list(range(1,50))


#subsetting just the odd ones
neighbors = list(filter(lambda x: x%2!=0, mylist))

#empty list that will hold cv scores
cv_scores = []

for k in neighbors:
	knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(knn, X_train,y_train, cv=10, scoring='accuracy')
	cv_scores.append(scores.mean())


MSE = [1 - x for x in cv_scores]


#determine best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("the optimal number of neighbors is %d" % optimal_k)


#plotting
plt.plot(neighbors,MSE)
plt.xlabel('number of neighbors')
plt.ylabel('Misclassification error')
plt.show()
