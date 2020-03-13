import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

##print(digits.data)
##print(digits.target)

clf = svm.SVC(gamma=0.0001, C=100)
X,y = digits.data[:-10], digits.target[:-10]
print(X)
print(y)

clf.fit(X,y)
print(clf.predict(digits.data[[-5]]))

plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
