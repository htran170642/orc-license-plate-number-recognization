from flask import Flask, jsonify,abort, request
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import keras
import image_processing
import base64
import time

import cnn
import svm 

app = Flask(__name__)
cnn_model = None
svm_model = None

def load_model():
    global cnn_model
    global svm_model
    cnn_model = cnn.load_model_cnn('models/weight.38-0.1598-0.9516.hdf5')	
    svm_model = svm.load_model_svm()

@app.route('/', methods=['POST', 'GET'] )
@cross_origin(origin='*')
def Hello():
    return 'Hello'

@app.route("/predict", methods=["POST"])
@cross_origin(origin='*')
def predict():
    if not request.json:
        flask.abort(400)

    data = {"success": False}
    
    imgbase64 = request.json['imgbase64']
    model_typ = request.json['model']
    start = time.process_time()
    if model_typ == 'cnn':   
        lp, roi, rmb = cnn.predict(cnn_model,str(imgbase64))
        if lp is not None:
            data['success'] = True
            data['lp'] = lp
            data['roi'] = roi
            data['rmb'] = rmb
    else:
        data['model'] = model_typ
        lp, roi, rmb = svm.predict(svm_model,str(imgbase64))
        if lp is not None:
            data['success'] = True
            data['lp'] = lp
            data['roi'] = roi
            data['rmb'] = rmb

    end = time.process_time() - start
    data['time'] = end
    
    return jsonify(data)

if __name__ == "__main__":
    load_model()
    # app.run(host="0.0.0.0", port=5000,debug = False, threaded = False)	
    app.run(debug = False, threaded = False)	

