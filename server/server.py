# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:29:20 2024

@author: phucn
"""
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import traceback
import pandas as pd
import joblib
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route("/", methods=['GET'])
@cross_origin()
def welcome():
    return ("Server is running...")
@app.route("/predict/lr", methods=['POST'])
@cross_origin()
def predictLr():
    if lr:
        try:
            json_data = request.get_json()
            data_frame = pd.DataFrame(json_data)
            print(json_data)
            data = transformer.transform(data_frame)
            print(data)
            prediction = list(lr.predict(data))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
@app.route("/predict/dt", methods=['POST'])
@cross_origin()
def predictDt():
    if dt:
        try:
            json_data = request.get_json()
            data_frame = pd.DataFrame(json_data)
            print(json_data)
            data = transformer.transform(data_frame)
            print(data)
            prediction = list(dt.predict(data))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
@app.route("/predict/rf", methods=['POST'])
@cross_origin()
def predictRf():
    if rf:
        try:
            json_data = request.get_json()
            data_frame = pd.DataFrame(json_data)
            print(json_data)
            data = transformer.transform(data_frame)
            print(data)
            prediction = list(rf.predict(data))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
if __name__ == '__main__':
    lr = joblib.load('../log_classifier.pkl')
    dt = joblib.load('../dt_classifier.pkl')
    rf = joblib.load('../rf_classifier.pkl')
    transformer = joblib.load('../pipeline_transform.pkl')
    print ('Model loaded')
    app.run(port=3000, debug=False)
       


