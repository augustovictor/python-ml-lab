from sklearn import tree

features = [
    [140, 1], # smooth
    [130, 1], # smooth
    [150, 0], # bumpy
    [170, 0] # bumpy
]
labels = [0, 0, 1, 1] #0 = apple, 1 = orange

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[160, 0]])) # 160 grams and bumpy