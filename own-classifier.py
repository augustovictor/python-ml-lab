from sklearn import datasets, tree
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

def euc(point_in_training_data, point_in_testing_data):
    return distance.euclidean(point_in_training_data, point_in_testing_data)

# Our own classifier
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data # features
y = iris.target # labels
"""
f(X) = y
"""

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5) #50%

# DecisionTree approach
# clf = tree.DecisionTreeClassifier()
# KNeighbors approach
# clf = KNeighborsClassifier()
# Our own classifier
clf = ScrappyKNN()

# Trainign
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))