import time
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)

import massageData
import utils

import tensorflow as tf

#Based on adopted codes from CS299 team 33

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_iter', 100, 'Number of steps to run trainer.')
flags.DEFINE_string('data_folder', 'data/numpy_bitmap_100/', 'Directory which has training data to use. Must have / at end.')
flags.DEFINE_string('results_folder', 'results/', 'Folder to store result outputs from run.')
flags.DEFINE_string('experiment_name', None, 'Name for the experiment. Useful to tagging files')

class_file_name = 'class_names_baselinev2_lr'
confusion_file_name = 'confusion_matrix_baselinev2_lr'

result = '' # printed result

def get_suffix_name():
    return "_" + FLAGS.experiment_name if FLAGS.experiment_name else ""

def get_class_filename():
    suffix_name = get_suffix_name()
    filename = "{}{}".format(class_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_confusion_matrix_name():
    return "{}{}".format(confusion_file_name, get_suffix_name())

def get_confusion_matrix_filename():
    suffix_name = get_suffix_name()
    filename  = "{}{}".format(confusion_file_name, suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def get_experiment_report_filename():
    suffix_name = get_suffix_name()
    filename =  "{}{}".format("baselinev2_lr_results", suffix_name)
    return os.path.join(FLAGS.results_folder, filename)

def kernel_svm(**args):
    global result
    print(args)
    dataGetter = massageData.massageData(folder=FLAGS.data_folder,samples = 2000) #train = 0.5, test=0.2
    X_train, Y_train = dataGetter.getTrain()
    X_dev, Y_dev = dataGetter.getDev()
    
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_dev = scaling.transform(X_dev)

    print("Starting " + args['kernel'] + " SVM training ...")
    start_time_secs = time.time()
    clf = svm.SVC(**args)
    clf.fit(X_train, Y_train)
    end_time_secs = time.time()
    print("Trained")

    training_duration_secs = end_time_secs - start_time_secs
    
    disp=[0,0]
    class_names = utils.get_label(Y_dev)
    file_name = "cm_svm_50_" + args['kernel']
    f = open('imgs/cm_imgs/'+file_name+'.txt','w')
    titles_options = [(file_name+"\nNormalized confusion matrix", 'true'),
                    (file_name+"\nConfusion matrix, without normalization", None)]
    for i,[title, normalize] in enumerate(titles_options):
        disp[i] = plot_confusion_matrix(clf, X_dev, Y_dev,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     xticks_rotation='vertical',
                                     include_values=False,
                                     normalize=normalize)
        disp[i].ax_.set_title(title)

        print(title,file=f)
        print(disp[i].confusion_matrix,file=f)
        
    #Y_dev_prediction = clf.predict(X_dev)
    accuracy = np.average(np.diagonal(disp[0].confusion_matrix))
    #accuracy = clf.score(X_dev, Y_dev)
    experiment_result_string = "-------------------\n"
    #experiment_result_string += "\nPrediction: {}".format(Y_dev_prediction)
    #experiment_result_string += "\nActual Label: {}".format(Y_dev)
    experiment_result_string += "\nAcurracy: {}".format(accuracy)
    experiment_result_string += "\nTraining time(secs): {}".format(training_duration_secs)
    print(experiment_result_string,file=f)
    result += experiment_result_string+'\n'

    confusion = disp[1].confusion_matrix
    #print ("Confusion matrix: ", confusion)
    pickle.dump(class_names, open("class_names_svm_l", 'wb'))
    pickle.dump(confusion, open("confusion_matrix_nclass_svm_l", 'wb'))
    #utils.create_confusion_matrices(class_names, confusion, file_name)
    
    #plt.subplot(2, 2, i+1)
    #plot_hyperplane(clf, X_dev, Y_dev, title=args['kernel'])
    #i += 1

def main():
    #plt.figure(figsize=(12,10),dpi=140)
    global result
    kernel_svm(kernel='poly', C=1.0, degree=5, coef0=1, verbose=1, max_iter=-1)
    kernel_svm(kernel='rbf', C=3.0, gamma='auto', verbose=1, max_iter=-1)
    #kernel_svm(kernel='sigmoid', C=2.0, coef0=2, verbose=1, max_iter=-1)
    #kernel_svm(kernel='linear', C=1.0, verbose=1, max_iter=-1)
    print(result)
    plt.show()

if __name__ == '__main__':
    main()
