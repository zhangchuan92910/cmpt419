import numpy as np
from math import log10, floor
from collections import defaultdict
import os

from utils import *

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def compute_distances(X1, X2, name="dists"):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))
    
    print("Computing Distances")
    if os.path.isfile(name+".npy"):
        dists = np.load(name+".npy")
    else:
        benchmark = int(round_to_1(M)//10)
        for i in range(len(X1)):
          for j in range(len(X2)):
            dists[i,j] = np.linalg.norm(X1[i] - X2[j])
          if i % benchmark == 0:
            print(str(i//benchmark)+"0% complete")
        np.save(name, dists)
    
    print("Distances Computed")
    return dists

def predict_labels(dists, y_train, y_val, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = defaultdict(list)

    for i in range(num_test):
        closest_y = y_train[np.argpartition(dists[i], k-1)[:k]]
        occur = np.bincount(closest_y)
        top = sorted(enumerate(occur), key=lambda a: a[1], reverse=True)
        
        y_pred[y_val[i]].append([cat for cat, _ in top[:3]])
    
    return y_pred

def predict_labels_weighted(dists, y_train, y_val, k=1):
    num_test, num_train = dists.shape
    y_pred = defaultdict(list)
    idx_to_cat, _ = get_category_mappings()

    for i in range(num_test):
        indices = np.argpartition(dists[i], k-1)[:k]
        closest = sorted([(y_train[j], dists[i][j]) for j in indices], key=lambda a: a[1])
        
        weights = np.linspace(1.0, 0.0, k)
        votes = defaultdict(float)
        for j in range(k):
            votes[closest[j][0]] += 1.0/np.sqrt(1.0*j+1.0) #1.0/closest[j][1] #1.0/np.sqrt(1.0*j+1.0) #weights[j]

        top = [(j, votes[j]) for j in sorted(votes, key=votes.get, reverse=True)]
        y_pred[idx_to_cat[y_val[i]]].append([idx_to_cat[cat] for cat, _ in top[:3]])
    
    return y_pred
