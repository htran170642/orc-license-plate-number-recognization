# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from image_processing import *

def load_model_cnn(path):
    return load_model(path)

def get_cnn_image_list(img, bounding_box):
    image_list = []
    for i in range(len(bounding_box)):
        tl, br = bounding_box[i]
        
        if img.ndim == 2: #grayscale image
            image  = img[tl[1]:br[1]+1, tl[0]:br[0]+1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            image_list.append(image)
            cv2.imwrite('cnn/'+str(i)+'.jpg', image)
        else: #ndim == 3
            image = img[tl[1]:br[1]+1, tl[0]:br[0]+1, :]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            image_list.append(image)
            cv2.imwrite('cnn/'+str(i)+'.jpg', image)

    return image_list


def predict(model, imgbase64):
    try:
        img = chuyen_base64_sang_anh(imgbase64.encode())
        img = resized_image(img)
        img_bin = cvt2bin(img)
        img_rmbound = remove_boundary(img_bin)
        cv2.imwrite('bbox/rmbox1.jpg', img_rmbound)
        bounding_box = get_box(img_rmbound)
        # img_resized, img_bin, edge, img_rmbound, bounding_box, type_list = preprocess(img)
        img_bbox = show_img_with_bbox(img, bounding_box)
        cv2.imwrite('bbox/bbox1.jpg', img_bbox)
        imgs_list = get_cnn_image_list(img, bounding_box)
        type_list = [True for i in range(len(imgs_list))]
        type_list[2] = False
        print(type_list)

        img_roi_path = "bbox/bbox1.jpg"
        cv2.imwrite(img_roi_path, img_bbox)
        roi_input =  open(img_roi_path, "rb")
        img_bbox = base64.b64encode(roi_input.read()).decode('utf-8')

        img_rmbound_path = "bbox/rmbox1.jpg"
        # cv2.imwrite(img_rmbound_path, img_rmbound * 255)
        cv2.imwrite(img_rmbound_path, img_rmbound)
        rmbound_input =  open(img_rmbound_path, "rb")
        rmbound_input = base64.b64encode(rmbound_input.read()).decode('utf-8')

    except:
        return None
    return get_prediction(model, imgs_list, type_list), img_bbox, rmbound_input


def get_prediction(model, imgs_list, type_list):
    ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9'}
    list_input = []
    list_out = []
    for image in imgs_list:
        if len(image.shape) == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (16, 32))
        image = np.expand_dims(image.astype('float32')/255, axis = 2)
        list_input.append(image)
    confidences = model.predict(np.array(list_input))
    for i, conf in enumerate(confidences):
        if not type_list[i]:
            pred = np.argmax(conf[:21])
            out = ALPHA_DICT[pred]
        else:
            pred = np.argmax(conf[21:]) + 21
            out = ALPHA_DICT[pred]
        list_out.append(out)
        # print(out)
    return list_out