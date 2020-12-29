#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn

from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from skimage.util import random_noise
from sklearn.cluster import KMeans
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix


# In[2]:


svm_NUM_DICT = {'0':0, '1':1, '2':2, '3':3, '4':4, '5': 5, '6':6, '7':7, '8':8, '9':9}
svm_ALPHA_DICT = {'A': ord('A'), 'B': ord('B'), 'C':ord('C'), 'D': ord('D'), 'E': ord('E'),
                  'F': ord('F'), 'G': ord('G'), 'H':ord('H'), 'K': ord('K'), 'L': ord('L'),
                  'M': ord('M'), 'N': ord('N'), 'P':ord('P'), 'R': ord('R'), 'S': ord('S'),
                  'T': ord('T'), 'U': ord('U'), 'V': ord('V'), 'X':ord('X'), 'Y': ord('Y'), 
                  'Z': ord('Z')
                 }


# In[3]:


svm_num_clusters = 150

svm_num_BoW = pickle.load(open('num_dictionary.pkl', 'rb'))
svm_num_pkl_filename = "num_pickle_model.pkl"
svm_num_loaded_model = pickle.load(open(svm_num_pkl_filename, 'rb'))

svm_alpha_BoW = pickle.load(open('alpha_dictionary.pkl', 'rb'))
svm_alpha_pkl_filename = "alpha_pickle_model.pkl"
svm_alpha_loaded_model = pickle.load(open(svm_alpha_pkl_filename, 'rb'))


# In[4]:


def svm_predict(images, isnum):
    def extract_sift_features(X):
        image_descriptors = []
        sift = cv2.SIFT_create()
        for i in range(len(X)):
            kp, des = sift.detectAndCompute(X[i], None)
            image_descriptors.append(des)
        return image_descriptors
    
    def create_features_bow(image_descriptors, BoW, num_clusters):
        X_features = []
        for i in range(len(image_descriptors)):
            features = np.array([0] * num_clusters)
            if image_descriptors[i] is not None:
                distance = cdist(image_descriptors[i], BoW)
                argmin = np.argmin(distance, axis=1)   
                for j in argmin:
                    features[j] += 1
            X_features.append(features)
        return X_features

    number_plate = []
    for i,img in enumerate(images):
        X = [img]
        if isnum[i]:
            BoW = svm_num_BoW
            model = svm_num_loaded_model
            dic = svm_NUM_DICT
        else:
            BoW = svm_alpha_BoW
            model = svm_alpha_loaded_model
            dic = svm_ALPHA_DICT
            
        image_descriptors = None
        X_features = None
        image_descriptors = extract_sift_features(X)
        X_features = create_features_bow(image_descriptors, BoW, num_clusters)
        number_plate.append(list(dic.keys())[list(dic.values()).index(model.predict(X_features))])
    return number_plate

# img = load_image('num/7/0_0118_00754_b.jpg.jpg')
# my_X = [img]

# my_image_descriptors = None
# my_X_features = None

# my_image_descriptors = extract_sift_features(my_X)
# my_X_features = create_features_bow(my_image_descriptors, BoW, num_clusters)

# loaded_model.predict(my_X_features)

