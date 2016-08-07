# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

# Let's consider our model in a function f(x) = y
# X <=> Features
# y <=> returned value / i.e. label
X = iris.data
y = iris.target

# Partition dataset into 2 parts
# 50% for training set (X_train)
# 50% for testing set (X_test)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

# Train the classifier
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# Display prediction for each row
#print predictions

#Check the accuracy of the model
# We comparte the predicted value with the true value
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)