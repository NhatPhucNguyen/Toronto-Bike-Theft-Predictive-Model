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
@app.route("/predict/lr", methods=['GET','POST'])
def predictLr():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            # return to data frame
            query = pd.DataFrame(scaled_df, columns=model_columns)
            print(query)
            prediction = list(lr.predict(query))
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
    model_columns = joblib.load('../model_columns.pkl')
    print ('Model columns loaded')
    app.run(port=3000, debug=True)


