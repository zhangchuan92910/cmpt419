import random
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import glob
import os
import sys
from collections import defaultdict
from utils import *
from k_nearest_neighbor import *
from sklearn.cluster import KMeans

def load_centroids():
    """
    Load centroids from centroids/npy directory.
    Returns map of category to its centroid, represented as numpy array.
    """
    if not os.path.isdir("centroids_plus_normalized") or\
            not os.path.isdir("centroids_plus_normalized/npy"):
        sys.exit("Need centroids/npy directory.")

    centroid_files = glob.glob("centroids_plus_normalized/npy/*.npy")
    centroids = np.zeros((len(centroid_files), len(np.load(centroid_files[0]))))
    labels = np.zeros(len(centroid_files))
    _, cat_to_idx = get_category_mappings()
    for i in range(len(centroid_files)):
        category = os.path.splitext(os.path.basename(centroid_files[i]))[0]
        centroids[i] = np.load(centroid_files[i])
        labels[i] = cat_to_idx[category.split('_')[0]]
    return centroids, labels

def group_centroids(centroids, labels):
    idx_to_cat, _ = get_category_mappings()
    kmeans = KMeans(n_clusters=70, random_state=0).fit(centroids)
    grouped_labels = kmeans.labels_
    group_to_category = defaultdict(list)
    category_to_group = {}
    for i in range(len(centroids)):
        group_to_category[grouped_labels[i]].append(labels[i])
        category_to_group[labels[i]] = grouped_labels[i]
    for i, group in group_to_category.items():
        print ("="*10)
        print ("GROUP ", i)
        for cat in group:
            print ("\t", idx_to_cat[int(cat)])
    return kmeans.cluster_centers_, group_to_category, category_to_group

def plot_scores(scores, accs, k, plot_accs=False):
    plt.figure()
    plt.plot(list(range(1, k+1)), scores, color='r', label='MAP@3')
    plt.plot(list(range(1, k+1)), accs, linestyle='dashed', color='b', label='MAP@1')
    plt.title('MAP@3 Score vs. k for KNN (K-Means++, Weighted by Rank)')
    plt.xlabel('k')
    plt.ylabel('MAP@3 Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x_train, y_train = load_centroids()
    #groups, group_to_category, category_to_group = group_centroids(x_train, y_train)
    x_val, y_val = load_dataset("test")
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    dists = compute_distances(x_val, x_train, "dists_test")
    
    f=open('Accuray.txt','a')
    print('===================================\n',file=f)
    print('# of classes:',len(y_train)//5,file=f)
    scores = []
    accs = []
    target = min(len(y_train),30)
    for k in range(1,target):

        pred = predict_labels_weighted(dists, y_train, y_val, k)
        #print(pred)
        if k<target-1:
            curr_score, acc = compute_scores(pred)
        else:
            curr_score, acc = compute_scores(pred,plott=True)
        print('k: ',k, 'score: ',curr_score,'accuracy:', acc/len(y_val),file=f)
        scores.append(curr_score)
        accs.append(acc/len(y_val))

    plot_scores(scores, accs, target-1)
