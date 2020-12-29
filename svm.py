import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
import sys
# sys.path.append('../')
from image_processing import *

from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from skimage.util import random_noise
from sklearn.cluster import KMeans
import random
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


# svm_num_pkl_filename = "svm/num_pickle_model.pkl"
# svm_num_loaded_model = pickle.load(open(svm_num_pkl_filename, 'rb'))

# svm_alpha_pkl_filename = "svm/alpha_pickle_model.pkl"
# svm_alpha_loaded_model = pickle.load(open(svm_alpha_pkl_filename, 'rb'))

def load_model_svm():
    svm_num_pkl_filename = "svm/num_pickle_model.pkl"
    svm_num_loaded_model = pickle.load(open(svm_num_pkl_filename, 'rb'))

    svm_alpha_pkl_filename = "svm/alpha_pickle_model.pkl"
    svm_alpha_loaded_model = pickle.load(open(svm_alpha_pkl_filename, 'rb'))
    return svm_num_loaded_model, svm_alpha_loaded_model


def predict(model, imgbase64):
    try:
        img = chuyen_base64_sang_anh(imgbase64.encode())
        img = resized_image(img)
        img_bin = cvt2bin(img)
        img_rmbound = remove_boundary(img_bin)
        # cv2.imwrite('bbox/rmbox1.jpg', img_bbox)
        bounding_box = get_box(img_rmbound)
        img_bbox = show_img_with_bbox(img, bounding_box)
        # cv2.imwrite('bbox/bbox1.jpg', img_bbox)
        images = get_image_list(img, bounding_box)
        type_list = [True for i in range(len(images))]
        type_list[2] = False

        img_roi_path = "bbox/bbox1.jpg"
        cv2.imwrite(img_roi_path, img_bbox)
        roi_input =  open(img_roi_path, "rb")
        img_bbox = base64.b64encode(roi_input.read()).decode('utf-8')

        img_rmbound_path = "bbox/rmbox1.jpg"
        cv2.imwrite(img_rmbound_path, img_rmbound)
        rmbound_input =  open(img_rmbound_path, "rb")
        rmbound_input = base64.b64encode(rmbound_input.read()).decode('utf-8')

    except:
        return None
    return svm_predict(model, images, type_list),img_bbox, rmbound_input

def get_image_list(img, bounding_box):
    image_list = []
    # print(img)
    for i in range(len(bounding_box)):
        tl, br = bounding_box[i]
        
        if img.ndim == 2: #grayscale image
            image  = img[tl[1]:br[1]+1, tl[0]:br[0]+1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            image_list.append(image)
            cv2.imwrite('svm/images/'+str(i)+'.jpg', image)
        else: #ndim == 3
            image = img[tl[1]:br[1]+1, tl[0]:br[0]+1, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            image_list.append(image)
            cv2.imwrite('svm/images/'+str(i)+'.jpg', image)

    return image_list

def svm_predict(svm_model, images, isnum):
    number_plate = []
    for i,im in enumerate(images):
        # X = [img]
        img = cv2.resize(im,(9,21),interpolation=cv2.INTER_AREA)
        if isnum[i]:
            # BoW = svm_num_BoW
            model = svm_model[0]
            dic = svm_NUM_DICT
        else:
            model = svm_model[1]
            dic = svm_ALPHA_DICT
        number_plate.append(list(dic.keys())[list(dic.values()).index(model.predict(np.asarray(img.ravel()).reshape(1,-1)))])
    # print(number_plate)
    return number_plate