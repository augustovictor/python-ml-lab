from sklearn import datasets, tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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
clf = KNeighborsClassifier()

# Trainign
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

