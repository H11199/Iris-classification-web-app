from sklearn import datasets
import numpy as np
import pickle

iris = datasets.load_iris()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)
knnModel = knn.fit(iris['data'],iris['target'])
pickle.dump(knnModel,open('iri.pkl','wb'))