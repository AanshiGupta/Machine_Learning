from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import decomposition#dimensionality reduction
from sklearn import datasets#predefined datasets


iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.55, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)
print(y_predicted)
print(y_test)
print(accuracy_score(y_test, y_predicted))



pca = decomposition.PCA(n_components=3)
pca.fit(X_train)
X_train_transformed = pca.transform(X_train)

clf2=svm.SVC()
clf2.fit(X_train_transformed, y_train)

X_test_transformed = pca.transform(X_test)#if train set is reduced to 3d then test set has also to be transformed to 3d
y_predicted_transformed = clf2.predict(X_test_transformed)
print(accuracy_score(y_predicted_transformed, y_test))

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
