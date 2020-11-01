"""program for feature extraction"""
"""author - karan and ashwin """
"""version_1.12"""
"""review - 2"""
import numpy as np
import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix  #import libraries for utilities
from sklearn.model_selection import train_test_split
dataset_folder = "C://Users//Karan's//speech-emotion-recognition-master//dataset"
class_labels = ["Neutral", "Angry", "Happy", "Sad"]
mslen = 32000  # Empirically calculated for the given dataset
def read_wav(filename):
    
    return wav.read(filename)       #function that returns tuple containing sampling frequency and signal
def get_data(flatten=True, mfcc_len=39):
    global data, labels
    data = []
    labels = []
    max_fs = 0                     #define required variables
    min_sample = int('9' * 10)
    s = 0
    cnt = 0
    cur_dir = os.getcwd()         #get the current working directory
    os.chdir('..')                 #goto home directory
    os.chdir(dataset_folder)       #change the pwd to the one with dataset
    for i, directory in enumerate(class_labels):
        #print("started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            max_fs = max(max_fs, fs)           
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mslen:
                pad_len = mslen - s_len        # pad the signals to have same size if lesser than required
               # pad_rem = pad_len % 2
               # pad_len /= 2
                signal = np.pad(signal, (pad_len, 0), 'constant', constant_values=0)
            else:
                pad_len = s_len - mslen
               # pad_rem = pad_len % 2            # else slice them
               # pad_len /= 2
                signal = signal[pad_len:pad_len + mslen]
            min_sample = min(len(signal), min_sample)
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)              #MFCC feature extraction

            if flatten:
                # Flatten the data
                mfcc = mfcc.flatten()     #normalise 
            data.append(mfcc)
            labels.append(i)
            cnt += 1
        #print ("ended reading folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)#split test and train data to 20 and 80 
    
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
def display_metrics(y_pred, y_true):
    #print("the accuracy score is ",accuracy_score(y_pred=y_pred, y_true=y_true))      #accuracy score 
    #print("the confusion matrix is ",confusion_matrix(y_pred=y_pred, y_true=y_true))    #display confusion matrix
    return accuracy_score(y_pred=y_pred, y_true=y_true),
