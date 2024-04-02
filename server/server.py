# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:29:20 2024

@author: phucn
"""
from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
app = Flask(__name__)
@app.route("/", methods=['GET'])
def welcome():
    return ("Server is running...")
@app.route("/predict/lr", methods=['POST'])
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
if __name__ == '__main__':
    lr = joblib.load('../log_classifier.pkl')
    print ('Model loaded')
    transformer = joblib.load('../pipeline_transform.pkl')
    app.run(port=3000, debug=False)
       


