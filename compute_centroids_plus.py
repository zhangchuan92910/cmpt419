import numpy as np
import glob
from matplotlib import pyplot as plt
import os
import sys
from collections import defaultdict
from utils import *
from sklearn.cluster import KMeans

def compute_centroids():
    """
    Computes centroid for each data file given.
    """
    centroids = {}
    cnts = defaultdict(int)
    idx_to_category, _ = get_category_mappings()
    
    train_examples = np.load("data/split/train_examples.npy")
    train_labels = np.load("data/split/train_labels.npy")
    labels=set(train_labels)
    examples_by_label={}
    for j in labels:
        examples_by_label[j] = train_examples[np.where(train_labels[:] == j)]

    #for i in range(train_examples.shape[0]):
    #    idx = train_labels[i]
    #    if train_labels[i] not in examples_by_label:
    #        examples_by_label[idx] = np.array(train_examples[i], dtype=np.float32)
    #    else:
    #        examples_by_label[idx] = np.append(examples_by_label[idx],\
    #                np.array(train_examples[i], dtype=np.float32))

    for category in labels:
        kmeans = KMeans(n_clusters=5, random_state=0).fit(examples_by_label[category])
        category = idx_to_category[int(category)]
        clusters = kmeans.cluster_centers_
        for a in range(5):
            name = category + "_" + str(a)
            centroids[name] = clusters[a]
        print("Done with category", category)
    
    return centroids

def create_centroids_dir():
    """
    Create centroids directory to save results.
    """
    for name in ["centroids_plus_normalized/","centroids_plus_normalized/npy","centroids_plus_normalized/png"]:
        try:
            os.makedirs(name)
        except OSError:
            continue # already exists

def save_centroids(centroids):
    """
    Save all images of centroids to centroids/png.
    Save all numpy arrays of centroids to centroids/npy.
    """
    for category, centroid in centroids.items():
        plt.imshow(np.reshape(centroid, (28, 28)), cmap='gray')
        plt.title(category)

        save_path = os.path.join("centroids_plus_normalized", category)
        plt.savefig("centroids_plus_normalized/png/"+category)
        np.save("centroids_plus_normalized/npy/"+category, centroid)
        # plt.show()

if __name__ == "__main__":
    if not os.path.isdir("data/split"):
        sys.exit("Need data directory.")
    centroids = compute_centroids()
    create_centroids_dir() 
    save_centroids(centroids)
    print("Done computing centroids!")
