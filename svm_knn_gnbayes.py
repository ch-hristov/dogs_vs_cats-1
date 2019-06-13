from utils import process_dataset
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


dataset_path = "./dogscats"
training_set = dataset_path + "/train"
validation_set =dataset_path + "/valid"

x_train, y_train, x_test, y_test = prepare_dataset(training_set, validation_set, size=100)

svm = SVC()
svm.fit(x_train, y_train)
score_svm = svm.score(x_test, y_test)
print(score_svm)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
score_knn = neigh.score(x_test, y_test)
print(score_knn)

gnbayes = GaussianNB()
gnbayes.fit(x_train, y_train)
score_gnbayes = gnbayes.score(x_test, y_test)
print(score_gnbayes)