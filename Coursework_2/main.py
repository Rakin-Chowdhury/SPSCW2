import utilities as u

import numpy as np
from matplotlib import pyplot as plt

d = load_data(train_set_path='data/wine_train.csv',
              train_labels_path='data/wine_train_labels.csv',
              test_set_path='data/wine_test.csv',
              test_labels_path='data/wine_test_labels.csv')
train = d[0]
label = d[1]
print(train)
'''
c1 = []
c2 = []
for j in range(76):
    for i in range(52):
        c1.append(train[i][j])
        c2.append(train[i][j])

plt.scatter(c1,c2)
plt.show()
'''
