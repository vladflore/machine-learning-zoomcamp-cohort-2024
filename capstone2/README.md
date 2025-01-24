# Capstone 2 project

The goal of this project is to train a model that can predict the type of waste in an image.

## Problem description

The dataset used originates from the [UC Irvine Machine Learning Repository Datasets](https://archive.ics.uci.edu/dataset/908/realwaste) and contains images of `9` different waste types. Given the dataset of images, the goal is to train a model to predict the type of waste in an image.

The waste types are:

- cardboard
- food organics
- glass
- metal
- miscellaneous trash
- paper
- plastic
- textile trash
- vegetation

## EDA

The dataset can be downloaded(normal download over Google Drive UI) from the following [Google Drive Link](https://drive.google.com/file/d/1VLgx5gONQ3RAeLSA2I6ylVaz0OjdwtQT/view?usp=sharing). The `zip` archive has a size of `657 MB`, and it contains `9` directories, each corresponding to a different type of waste. Each directory contains images of the respective waste type.

Feel free to have a look in the [notebook](./notebook_colab.ipynb) for more details about the dataset.

Before working with the dataset, please follow the steps below:

1. Download the dataset from the provided link to a location of your choosing (e.g. `~/Downloads`)
2. Clone this repository to a preferred location, and navigate to the `capstone2` directory.

```shell
git clone git@github.com:vladflore/machine-learning-zoomcamp-cohort-2024.git
cd machine-learning-zoomcamp-cohort-2024/capstone2
```

3. Unzip the dataset

Run the following in the `capstone2/` directory.

```shell
rm -rf content # remove the content directory if it exists
mkdir -p content/drive/mydrive/mlzoomcamp/capstone2 # create the directory where the dataset will be stored
mv ~/Downloads/realwaste.zip .
unzip -o realwaste.zip -d content/drive/mydrive/mlzoomcamp/capstone2
```

The dataset is structured as follows:

```shell
tree -L 1 content/drive/mydrive/mlzoomcamp/capstone2/materials/
content/drive/mydrive/mlzoomcamp/capstone2/materials/
├── Cardboard
├── Food Organics
├── Glass
├── Metal
├── Miscellaneous Trash
├── Paper
├── Plastic
├── Textile Trash
└── Vegetation
```

## Exporting notebook to script

Note that the notebook is actually a Google Colab notebook, and the exported scripts have been adjusted accordingly to run on a local machine.

Two scripts have been created from the notebook:

- `prep.py` renames the folders for each waste type to lowercase and removes the white spaces; additionally, splits the dataset into training, validation and test sets
- `train.py` contains the code for training the model, which is then saved to a file

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
pip install -r requirements.txt
```

## Reproducibility

Once the environment is set up, the scripts can be run in the following order:

```shell
python prep.py
python train.py
```

### Running `prep.py`

The script does the following:

- rename each waste type folder to lowercase, replacing spaces with `_`
- shows the total number of images in the dataset, in this case `4752`
- creates three subsets, `train`, `val` and `test`, with `2848` (`60%`), `950` (`20%`), and `954` (`20%`) images, respectively

The images for training the model are structured as follows:

```shell
tree -L 3 content/drive/mydrive/mlzoomcamp/capstone2/dataset
content/drive/mydrive/mlzoomcamp/capstone2/dataset
├── test
│   └── materials
│       ├── cardboard
│       ├── ...
├── train
│   └── materials
│       ├── cardboard
│       ├── ...
└── val
    └── materials
        ├── cardboard
        ├── ...
```

### Running `train.py`

The script builds a CNN, trains it and saves the best version to a file. The parameters used in the training are:

```python
learning_rate = 0.001
size_inner = 1000
droprate = 0.2
```

These have been chosen based on the run in the notebook. Have a look at the notebook for more details.

This is a long process, and it is recommended to run it on a machine with a GPU.

The images are resized to `(299, 299)` and the model is trained for `50` epochs.

At the end, a file like `xception_31_0.844.keras` is saved, which represents the best model obtained during the training. In this case, the model has an accuracy of `84.4%` on the validation set.

> **Note:** If you choose not to run this file, you can download the model from the following [Google Drive Link](https://drive.google.com/file/d/1N_Z2JDAOj2k5BSABXga8Q5JrLrnLCzNF/view?usp=sharing).

## Deployment

The model is deployed as a lambda function. The code for the lambda function is in the `lambda_function.py` file.

The notebook `notebook_colab.ipynb` contains the code for preparing the model for deployment, i.e. converting from `keras` to `tflite`, and making use of the `tflite` model in the lambda function.

> **Note:** the converted model, i.e. the `.tflite` file, can be downloaded from [this link](https://drive.google.com/file/d/1PMTUlqSSgzNrFNPBcNC2h8z5xOrLYYwY/view?usp=drive_link). This file is needed for the containerization part. 

## Containerization

The `Dockerfile` contains the necessary instructions for building the image.

> **Note:** Before continuing, make sure the model (`.tflite`) is downloaded and placed in the `capstone2` directory.

### Building the image

```shell
# Build the image
docker build -t capstone-model-predict-materials .
```

### Publishing the image

```shell
# Tag the image
docker image tag capstone-model-predict-materials:latest vladflore/mlzoomcamp-capstone2:latest
# Push the image to Docker Hub
docker push vladflore/mlzoomcamp-capston2:latest
```

### Running the container

```shell
# Pull the image from Docker Hub
docker pull vladflore/mlzoomcamp-capstone2:latest
# Run the container
docker run -it --rm -p 8080:8080 vladflore/mlzoomcamp-capstone2:latest
```

To run the local image, instead of the Docker Hub one, use the following command:

```shell
docker run -it --rm -p 8080:8080 capstone-model-predict-materials
```

## Cloud deployment

There is no cloud deployment for this project.

## Showcase

The script `test.py` can be used to test the lambda function locally. The script takes an image URL as an argument and sends a POST request to the lambda function. Make sure the lambda function is running locally before running the script, i.e. the container is running.

```shell
python test.py --image-url https://raw.githubusercontent.com/vladflore/machine-learning-zoomcamp-cohort-2024/refs/heads/main/capstone2/Cardboard_1.jpg
```

The output should be similar to the following:

```json
{ "material": "cardboard" }
```

[This cast](https://asciinema.org/a/ursLOwpYwNQYgd5Y901UNXmP9) shows three prediction runs, two of them correctly predicting the material, while the third confuses glass with plastic.

## Files overview

These files are already present in the repository.

```shell
tree -L 1 .
.
├── Cardboard_1.jpg # test image
├── Dockerfile # instructions for building the image
├── Glass_13.jpg # test image
├── Glass_5.jpg # test image
├── README.md # this file
├── lambda_function.py # code for the lambda function
├── notebook_colab.ipynb # Google Colab notebook
├── prep.py # script for preparing the dataset
├── requirements.txt # dependencies
├── test.py # script for testing the lambda function
├── train.py # script for training the model
```

Additionally, the following files have to be downloaded:

- `realwaste.zip` - the dataset
- `xception_31_0.844.tflite` - the model
