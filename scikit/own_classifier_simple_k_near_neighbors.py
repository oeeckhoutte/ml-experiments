from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a, b)

# For now we avoid the K variable. We suppose k = 1
# That's why this class is called Scrappy...
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
			if(dist < best_dist):
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

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

# K Near Neighbor algo
#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()
	

# Train the classifier
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
# Display prediction for each row
#print predictions

#Check the accuracy of the model
# We comparte the predicted value with the true value
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)