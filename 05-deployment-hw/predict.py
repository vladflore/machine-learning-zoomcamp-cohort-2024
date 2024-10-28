import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask
from flask import request

app = Flask('prediction-service')
model_file = "model1.bin"
dv_file = "dv.bin"


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    with open(model_file, 'rb') as model_file_in:
       model = pickle.load(model_file_in)

    with open(dv_file, 'rb') as dv_file_in:
       dv = pickle.load(dv_file_in)

    #customer = {"job": "management", "duration": 400, "poutcome": "success"}
    X = dv.transform([customer])
    prediction = model.predict_proba(X)[0,1]
    #print(prediction)
    result = {'prediction': float(prediction)}

    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
