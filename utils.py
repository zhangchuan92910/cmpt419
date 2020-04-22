import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
import itertools
import os
import sys
from collections import Counter
from sklearn.metrics import plot_confusion_matrix
from collections import defaultdict

# Based on adopted codes from CS299 team 33 and 98

# Dimension for the input feature bitmap vectors.
BITMAP_DIM = 784

def get_label(Y):
    labels = list(set(Y))
    print ('Labels: ', labels)
    return labels

def create_confusion_matrices(class_names, confusion, file_name):

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(confusion,classes=class_names,
                                     include_values=False,
                                    title='Confusion matrix, without normalization')
    #fig1.savefig('imgs/cm_imgs/' + file_name + '.png')
    plt.plot()
    f = open('imgs/cm_imgs/' + file_name + '.txt','w')
    print('Confusion matrix, without normalization',file=f)
    print(confusion,file=f)
    f.close()
    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(confusion, classes=class_names, normalize=True,
                                     include_values=True,
                                     title='Normalized confusion matrix')
    #fig2.savefig('imgs/cm_imgs/' + file_name + '_norm.png')
    plt.plot()
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          include_values=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if include_values:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def write_contents_to_file(output_file, input_string):
    with open(output_file, 'w') as file_handle:
        file_handle.write(input_string)
        
def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def load_dataset(dataset):
    """Load dataset, one of train, val, or test."""

    if dataset not in ["train", "val", "test", "train_normalized", "val_normalized"]:
        sys.exit("Invalid dataset type.")
    x_path = "data/split/{}_examples.npy".format(dataset)
    y_path = "data/split/{}_labels.npy".format(dataset.split('_')[0])
    
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        sys.exit("Missing dataset files.")
    
    return np.load(x_path), np.asarray(np.load(y_path), dtype=np.int32)

def get_category_mappings(fname = 'category_list.txt'):
    """
    Arg: filename fname
    Returns: Mapping from index to category and mapping from category to index
    """
    with open(fname) as f:
        content = f.readlines()
    category_to_index = dict()
    index_to_category = []
    for i in range(len(content)):
        category_name = content[i].strip()
        index_to_category.append(category_name)
        category_to_index[category_name] = i
    return index_to_category, category_to_index

def get_groupings(per_category_mapk, num=2):
    '''Create groupings of categories based on common guesses.
    Num is the number of guesses to take into account.'''
    index_to_category, _ = get_category_mappings()
    assignments = {}
    groupings = defaultdict(set)

    for category, acc, guess in per_category_mapk:
        assignment = -1
        if category not in assignments:
            for cat, _ in guess[:num]:
                if cat in assignments:
                    assignment = assignments[cat]
                    break
            if assignment == -1:
                assignment = len(groupings)
            assignments[category] = assignment
            groupings[assignment].add(category)
        else:
            assignment = assignments[category]
        for cat, _ in guess[:num]:
            if cat not in assignments:
                assignments[cat] = assignment
                groupings[assignment].add(cat)

    print(assignments)
    for i, grouping in groupings.items():
        print("="*10)
        print("GROUP", i)
        for g in grouping:
            print("\t", g)


def plot_accuracies(acc_vals):
    '''Plot histogram of MAP@3 Values'''
    plt.figure("KNN Accuracies")
    plt.hist(acc_vals, 19)
    plt.title("MAP@3 Accuracy Distribution for KNN (K-Means++, Weighted)")
    plt.xlabel("MAP@3 Accuracy")
    plt.ylabel("Number of Categories with Given Accuracy")
    plt.savefig("KNN_Accuracies")

def compute_scores(pred, verbose=False, plott=False):
    actual, predicted, per_category_mapk = [], [], []
    total_accuracy = 0.0
    for category, guesses in pred.items():
        cur_actual, cur_predicted = [], []
        occ = Counter()
        for guess in guesses:
            cur_actual.append([category])
            cur_predicted.append(guess)
            for cat in guess:
                occ[cat] += 1
            if guess[0] == category:
                total_accuracy += 1
        per_category_mapk.append((category, mapk(cur_actual, cur_predicted), occ.most_common(3)))
        actual += cur_actual
        predicted += cur_predicted
    per_category_mapk.sort(key=lambda x: -x[1])

    # Get MAP@3 scores for all categories
    index_to_category, _ = get_category_mappings()
    acc_vals = []
    for category, acc, guess in per_category_mapk:
        print(category, "MAPK@3:", acc, "common guesses:",[(g[0], g[1]) for g in guess])
        acc_vals.append(acc)
    
    # Get groupings of categories based on common guesses
    get_groupings(per_category_mapk)

    # Plot histogram of accuracies
    if plott:
        plot_accuracies(acc_vals)

    if verbose:
        print("="*30)
        print("MAPK@3:", mapk(actual, predicted))
        print("TOTAL ACCURACY:", total_accuracy/len(actual))
        for category, acc, guess in per_category_mapk:
            print(category, "MAPK@3:", acc, "common guesses:", guess)

    return mapk(actual, predicted), total_accuracy