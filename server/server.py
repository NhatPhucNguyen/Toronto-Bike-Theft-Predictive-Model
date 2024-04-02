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
            print(data_frame.dtypes)
            categorical_features = []
            for col, col_type in data_frame.dtypes.items():
                 if col_type == 'O':
                      categorical_features.append(col)
            query = pd.get_dummies(data_frame,columns=categorical_features,dummy_na=False)
            print(query)
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query['OCC_YEAR'])
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            # Fit your data on the scaler object
            scaled_df = scaler.fit_transform(query)
            #print(scaled_df)
            # return to data frame
            data = pd.DataFrame(scaled_df,columns=model_columns)
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
    model_columns = joblib.load('../model_columns.pkl')
    print ('Model columns loaded')
    app.run(port=3000, debug=False)
       


