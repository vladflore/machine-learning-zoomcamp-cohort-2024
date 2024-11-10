import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from flask import Flask
from flask import request

app = Flask("prediction-service")
model_file = "model.bin"


@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()

    with open(model_file, "rb") as model_file_in:
        dv, model = pickle.load(model_file_in)

    X = dv.transform([patient])
    prediction = model.predict_proba(X)[:, 1]
    result = {"prediction": float(prediction), "stroke": bool(prediction >= 0.5)}

    return result


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
