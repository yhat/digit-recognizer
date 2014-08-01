import numpy as np
import pandas as pd
from PIL import Image
from StringIO import StringIO
import base64
import os
import csv
from ggplot import *
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split

wd = "./numbers/"
files = [f for f in os.listdir(wd)]
files = [wd + f for f in files]

STANDARD_SIZE = (50, 50)
def get_image_data(filename):
    img = Image.open(filename)
    img = img.getdata()
    img = img.resize(STANDARD_SIZE)
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

data = []
labels = []
print "extracting features..."
for i, f in enumerate(files):
    print i, "of", len(files)
    data.append(get_image_data(f))
    labels.append(int(f.split(".")[-2][-1]))
print "done.\n"


output = open('./results.csv', 'w')
w = csv.writer(output)
w.writerow(["actual"] + range(10))
results = []
for n_components in [2, 5, 10, 25, 50, 100, 250, 500, 1000]:
    print "Training for %d components..." % n_components
    pca = RandomizedPCA(n_components=n_components)
    std_scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=13)
    clf.fit(X_train, y_train)
    print "Results for %d components:" % n_components
    cm = confusion_matrix(y_test, clf.predict(X_test))

    for i, row in enumerate(cm):
        w.writerow([i] + row.tolist())

    acc = accuracy_score(y_test, clf.predict(X_test))
    # print precision_score(y_test, clf.predict(X_test))
    # print recall_score(y_test, clf.predict(X_test))
    print acc
    results.append({"n_components": n_components, "accuracy": acc})
output.close()


results = pd.DataFrame(results)
print ggplot(results, aes(x='n_components', y='accuracy')) + \
    geom_line()

