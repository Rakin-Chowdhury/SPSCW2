#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
#from voronoi import plot_voronoi

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'


MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']


def feature_selection(train_set, train_labels, **kwargs):

    #n_features = train_set.shape[1]
    #fig, ax = plt.subplots(n_features, n_features)
    #plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)

    #ax2 = plt.subplots(1,1)

    Class_Colours= [CLASS_1_C, CLASS_2_C, CLASS_3_C]
    Colours = []

    #print(train_labels)

    #for i in train_labels:
        #print(i)
        #Colours.append(Class_Colours[int(i)-1])

    #print(Colours)

    #for j in range(13):
        #for k in range(13):
            #ax[j,k].scatter(train_set[:, j],train_set[:, k], c = Colours, s = 3)
            #ax[j,k].set_title('F {} x {}'.format(j+1, k+1))
    #plt.scatter(train_set[:, 9],train_set[:, 6], c = Colours)
    #plt.xaxis.label.set_size(1)
    #plt.yaxis.label.set_size(1)
    #plt.labelsize(1)
    #plt.show()
    fe = np.array([6, 9])
    


    return [fe]

def centroid(train_set_2, train_labels):
    index=np.unique(train_labels)
    roid = np.array([np.mean(train_set_2[train_labels == c,:], axis = 0) for c in index])
    return roid, index



def knn(train_set, train_labels, test_set, k, **kwargs):
    train_set_2 = train_set[:, [6,9]]
    """index=np.unique(train_labels)"""
    """print(centroid(train_set_2,train_labels))"""
    index = centroid(train_set_2, train_labels)[1]
    coord = centroid(train_set_2, train_labels)[0]
    plt.scatter(train_set_2[:,0],train_set_2[:,1])
    for x in range(len(index)):
        plt.scatter(coord[x][0],coord[x][1], s=600)
    plt.show()
    return []


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
