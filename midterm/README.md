# Midterm project

The goal of this project is to predict whether a patient is likely to have a stroke based on the provided features.

## Problem description

Dataset used: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

This dataset is used to predict whether a patient is likely to get a stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.

The dataset contains the following features:

- `id`: unique identifier
- `gender`: gender of the patient
- `age`: age of the patient
- `hypertension`: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- `heart_disease`: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- `ever_married`: patient's marital status
- `work_type`: patient's employment status
- `Residence_type`: patient's residence type
- `avg_glucose_level`: average glucose level in blood
- `bmi`: body mass index
- `smoking_status`: patient's smoking status
- `stroke`: 1 if the patient had a stroke or 0 if not

The target variable, what we want to predict, is the `stroke` column.

## EDA

A couple of notable observations discovered during the EDA:

- the dataset is imbalanced, with only 4.9% of the patients having experienced a stroke
- the dataset contains missing values in the `bmi` column
- the dataset seems to be ordered by the `stroke` column, with the patients that had a stroke being at the beginning of the dataset
- the dataset contains both numerical and categorical features
- there is one patient that has a `gender` value of `Other`
- the strongest correlations are between: `age` and `bmi` (0.325942), `age` and `hypertension` (0.276398), `age` and `heart_disease` (0.263796)
- some weaker correlations include: `bmi` and `stroke` (0.038947), `bmi` and `heart_disease` (0.038899)
- `ever_married` and `work_type` have moderate dependency, judged by the mutual information score, others have rather low dependency

## Model training

Two models have been trained and evaluated: a _logistic regression_ model and a _gradient boosting_ model.

The _gradient boosting_ model yields an AUC score of `0.8003` on the validation set.

The _logistic regression_ model yields an AUC score of `0.8088` on the validation set.

```python
LogisticRegression(C=10, max_iter=1000, random_state=42, solver='liblinear')
```

```python
final_eta = 0.3
final_max_depth = 4
final_min_child_weight = 1
final_num_boost_round = 10
xgb_params = {
    'eta': final_eta,
    'max_depth': final_max_depth,
    'min_child_weight': final_min_child_weight,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}
model = xgb.train(xgb_params, dtrain, num_boost_round=final_num_boost_round)
```

The final model chosen for deployment is the _logistic regression_ model, which yields an AUC score of `0.8526` on the test set.

## Exporting notebook to script

The `train.py` script is responsible for training the machine learning model using the provided dataset and saving the trained model to a file. The `predict.py` script loads the saved model and uses it to make predictions based on new input data. Additionally, `predict.py` exposes a `POST` endpoint that accepts a JSON payload and returns the prediction.

## Reproducibility

The dataset is stored in the file `healthcare-dataset-stroke-data.csv`. Running `python train.py` will train the model and save it to the file `model.bin`. Running `python predict.py` will load the model and start a Flask server that listens on port `9696`.

Predictions can be made by issuing:

```shell
curl -X POST -H 'Content-type: application/json' --data @person.json http://localhost:9696/predict
```

The payload comes from the `person.json` file, which contains the patient's information in JSON format.

## Deployment

Running `python predict.py` will load the model and start a Flask server that listens on port `9696`.

## Dependency and environment management

All the dependencies are stored in the `Pipfile` and `Pipfile.lock` files. The environment can be recreated by running `pipenv install` in the project directory.

If `pipenv` is not installed, it can be installed by running `pip install pipenv`.

Steps:

- change to the project directory: `cd midterm`
- create a virtual environment: `pipenv --python 3.11`
  One of the output lines should mention the location of the virtual environment, e.g. `Virtualenv location: /Users/<your-username>/.local/share/virtualenvs/midterm-WC4oPiq4`
- install the dependencies: `pipenv install`

Once the environment is created, it can be activated by running `pipenv shell`.

Run the scripts: `python train.py` and `python predict.py`.

To remove the environment, run `pipenv --rm`.

## Containerization

The project has been containerized using Docker. The `Dockerfile` contains the necessary instructions to build the image and run the container.

### Building the image

```shell
docker build -t stroke-prediction .
```

### Running the container

```shell
docker run -it -p 9696:9696 stroke-prediction:latest
```

A public pre-built image is available on Docker Hub: [https://hub.docker.com/r/vladflore/mlzoomcamp-midterm](https://hub.docker.com/r/vladflore/mlzoomcamp-midterm)

```shell
docker pull vladflore/mlzoomcamp-midterm:latest
```

This can be run in the usual manner: `docker run -it -p 9696:9696 vladflore/mlzoomcamp-midterm:latest`

Tagging and pushing the image to Docker Hub can be done with the following commands:

```shell
docker image tag stroke-prediction:latest vladflore/mlzoomcamp-midterm:latest

docker push vladflore/mlzoomcamp-midterm:latest
```

Prerequisites:

- Docker installed
- Docker Hub account
- Logged in to Docker Hub
- Docker Hub repository created

## Cloud deployment

There is no cloud deployment for this project.

## Showcase

This cast shows the steps to create the virtual environment, install the dependencies, train the model, and run the Flask server: [cast 1](https://asciinema.org/a/YNjIhC5EPLc1a3iBl78q2cTSU)

This cast shows the interaction with the deployed model, run locally: [cast 2](https://asciinema.org/a/RBqsqQcaqp3vkDuMnWEfvtCVP)
