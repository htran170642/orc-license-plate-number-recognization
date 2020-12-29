import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
import base64
from char_splitting import *

PATH = 'saved_images'

def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64

def resized_image(img):
    rate = 80 / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=rate, fy=rate)
    return img

def cvt2bin(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return img_bin

def remove_boundary(img):
    img2 = img.copy()
    img2 = cv2.morphologyEx(img2,
                            cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3,1)))
    img2[:10, :10] = 0
    contours, _ = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img3 = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x in (0, img2.shape[1]-w) or y in (0, img2.shape[0]-h):
            cv2.drawContours(img3, [contour], 0, (0, 0, 0), -1)
        
    img3[:10, :10] = 0
    return img3

def get_box(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 0.015 * img.size: #small noisy elements
#             print("Small element at {}, {}, {:.2%} total area".format(x, y, w*h/img.size))
            continue
        bounding_box.append(((x, y), (x+w-1, y+h-1)))
    bounding_box = sorted(bounding_box, key=lambda b: b[0][0] + b[0][1]*3)
    return bounding_box

def show_img_with_bbox(img, bounding_box):
    # fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    # ax = ax.ravel()
    img_show = img.copy()
    for i in range(len(bounding_box)):
        tl, br = bounding_box[i]
        img_show = cv2.rectangle(img_show, tl, br, (0, 255, 0))
    return img_show


def show_characters(img, bounding_box):
    for i in range(len(bounding_box)):
        tl, br = bounding_box[i]
        if img.ndim == 2: #grayscale image
            cv2.imwrite(PATH+'/'+str(i)+'.jpg', img[tl[1]:br[1]+1, tl[0]:br[0]+1])
        else: #ndim == 3
            cv2.imwrite(PATH+'/'+str(i)+'.jpg', img[tl[1]:br[1]+1, tl[0]:br[0]+1, :])
