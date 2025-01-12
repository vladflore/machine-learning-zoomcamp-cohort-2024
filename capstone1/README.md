# Capstone 1 project

The goal of this project is to train a model to be used in a lambda function that predicts the type of animal in an image.

## Overview of the files in the project

```shell
tree -L 1 .
.
├── Dockerfile # Dockerfile for containerization
├── README.md # this file
├── butterfly.jpeg # test image
├── cat.jpeg # test image
├── content # directory where the dataset is stored (is to be created)
├── dog.jpeg # test image
├── histo.png # histogram of the number of images per animal type
├── lambda_function.py # code for the lambda function
├── notebook_colab.ipynb # Google Colab notebook
├── pre_deployment_prep.ipynb # notebook for preparing the model for deployment
├── prep.py # script for preparing the dataset
├── raw-img-20250111T215411Z-001.zip # dataset archive (is to be downloaded from Google Drive)
├── test-model.cast # asciinema cast for testing the lambda function
├── test.py # script for testing the lambda function
├── test_model.py # script for testing the model
├── to_skip.txt # text file with images that are not jp(e)g (is to be created)
├── top_ten.png # pie chart of the top 10 image sizes in the dataset
├── train.py # script for training the model
├── translate.py # script for translating the animal names from Italian to English
├── xception_08_0.977.keras # best model obtained during training
└── xception_08_0.977.tflite # best model converted to tflite format
```

## Problem description

This dataset used originates from the [Kaggle dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data) and contains images of `10` different types of animals. Given the dataset of images, the goal is to train a model to predict the type of animal in an image.

## EDA

The dataset can be downloaded(normal download over Google Drive UI) from the following [Google Drive Link](https://drive.google.com/file/d/121CqZuMGq2mWpo3IXgXVuhex4AHTEG-G/view?usp=drive_link). The `zip` archive has a size of `598 MB`, and it contains `10` directories, each corresponding to a different type of animal. Each directory contains images of the respective animal.

Before working with the dataset, please follow the steps below:

1. Download the dataset from the provided link.
2. Clone this repository to a preferred location, and navigate to the `capstone1` directory.

```shell
git clone git@github.com:vladflore/machine-learning-zoomcamp-cohort-2024.git
cd machine-learning-zoomcamp-cohort-2024/capstone1
```

3. Unzip the downloaded archive.

```shell
rm -rf content # remove the content directory if it exists
mkdir -p content/drive/mydrive/mlzoomcamp/capstone1/animals # create the directory where the dataset will be stored
unzip -o raw-img-20250111T215411Z-001.zip -d content/drive/mydrive/mlzoomcamp/capstone1/animals
```

The dataset is structured as follows:

```shell
tree -L 1 content/drive/mydrive/mlzoomcamp/capstone1/animals/raw-img
content/drive/mydrive/mlzoomcamp/capstone1/animals/raw-img
├── cane
├── cavallo
├── elefante
├── farfalla
├── gallina
├── gatto
├── mucca
├── pecora
├── ragno
└── scoiattolo
```

Note that the names of the animals are in Italian and that there is no splitting of the dataset into training, validation and test sets. The file `translate.py` provides the translation of the animal names from Italian to English.

For a more detailed insight into the dataset, please refer to the section `Reproducibility -> Running prep.py`.

## Model training

The model is trained using a Convolutional Neural Network(CNN). The model is built using the `Keras` library, and the `Xception` architecture is used as the base model. The model is trained using the `Adam` optimizer and the `CategoricalCrossentropy` loss function. See `train.py` for more details.

## Exporting notebook to script

The notebook `notebook_colab.ipynb` has been exported to two scripts: `prep.py` and `train.py`.

Note that the notebook is actually a Google Colab notebook, and the exported scripts have been adjusted accordingly to run on a local machine.

`prep.py` contains some exploratory code, together with splitting the dataset into training, validation and test sets. Additionally, since the datasets are rather large, three smaller subsets are created to be used in the training and testing of the model.

`train.py` contains the code for training the model.

## Reproducibility

Running the scripts locally requires setting up a virtual environment and installing the necessary dependencies(see below). Once the environment is set up, the scripts can be run in the following order:

```shell
python prep.py
python train.py
```

### Running `prep.py`

The script produces the following:

- total number of images in the dataset, in this case `26236`
- display of 18 randomly selected images from each of the animal types
- a text file with images that are not `jp(e)g`, `to_skip.txt`, which contains `60` entries
- a [histogram](./histo.png) of the number of images per animal type
- a [pie chart](./top_ten.png) of the top 10 image sizes in the dataset
- three subsets of the dataset, `train`, `val` and `test` per animal type

```shell
Total number of usable images in the dataset: 26176

gallina: train=1858, val=620, test=620
ragno: train=2892, val=964, test=965
gatto: train=1000, val=333, test=334
farfalla: train=1239, val=413, test=414
mucca: train=1119, val=373, test=374
cavallo: train=1573, val=525, test=525
cane: train=2929, val=977, test=977
pecora: train=1091, val=364, test=364
scoiattolo: train=1117, val=372, test=373
```

The smaller datasets for training the model are structured as follows:

```shell
tree -L 1 content/drive/mydrive/mlzoomcamp/capstone1/animals/small
content/drive/mydrive/mlzoomcamp/capstone1/animals/small
├── test
├── train
└── val
```

### Running `train.py`

The script builds a CNN, trains it and saves the best version to a file. The parameters used in the training are:

```python
learning_rate = 0.0001
size_inner = 1000
droprate = 0.8
```

These have been chosen based on the run in the notebook. Have a look at the notebook for more details.

This is a long process, and it is recommended to run it on a machine with a GPU.

The images are resized to `(299, 299)` and the model is trained for `50` epochs.

At the end, a file like `xception_08_0.977.keras` is saved, which represents the best model obtained during the training. In this case, the model has an accuracy of `0.977` on the validation set.

## Deployment

The model is deployed as a lambda function. The code for the lambda function is in the `lambda_function.py` file. The notebook `pre_deployment_prep.py` contains the code for preparing the model for deployment.

## Dependency and environment management

Running the scripts locally requires setting up a virtual environment and installing the necessary dependencies. The environment can be set up following the steps below.

The python version used is `3.9.20`. Make sure to use the correct version when creating the virtual environment.

```shell
/opt/anaconda3/envs/ml-zoomcamp/bin/python -V
# Python 3.9.20
```

```shell
# create the virtual environment
/opt/anaconda3/envs/ml-zoomcamp/bin/python -m venv .venv
# activate the virtual environment
source .venv/bin/activate
# install the dependencies
pip install tensorflow==2.17.1 keras==3.5.0 scipy matplotlib
```

## Containerization

The `Dockerfile` contains the necessary instructions for building the image.

### Building the image

```shell
# Build the image
docker build -t capstone-model-predict-animals .
```

### Publishing the image

```shell
# Tag the image
docker image tag capstone-model-predict-animals:latest vladflore/mlzoomcamp-capstone1:latest
# Push the image to Docker Hub
docker push vladflore/mlzoomcamp-capstone1:latest
```

### Running the container

```shell
# Pull the image from Docker Hub
docker pull vladflore/mlzoomcamp-capstone1:latest
# Run the container
docker run -it --rm -p 8080:8080 vladflore/mlzoomcamp-capstone1:latest
```

To run the local image, instead of the Docker Hub one, use the following command:

```shell
docker run -it --rm -p 8080:8080 capstone-model-predict-animals
```

## Cloud deployment

There is no cloud deployment for this project.

## Showcase

The script `test.py` can be used to test the lambda function locally. The script takes an image URL as an argument and sends a POST request to the lambda function. Make sure the lambda function is running locally before running the script, i.e. the container is running.

```shell
python test.py --image-url https://raw.githubusercontent.com/vladflore/machine-learning-zoomcamp-cohort-2024/refs/heads/main/capstone1/cat.jpeg
```

The output should be similar to the following:

```json
{ "animal": "cat" }
```

Check this cast [here](https://asciinema.org/a/DuEeaLZY1jdVf3Z0eCyOgRtMq).
