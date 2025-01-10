# Capstone 1 project

The goal of this project is to predict ...

## Problem description

Dataset used:

This dataset is used to predict ...

https://drive.google.com/file/d/121CqZuMGq2mWpo3IXgXVuhex4AHTEG-G/view?usp=drive_link

## EDA

A couple of notable observations discovered during the EDA:

...

## Model training

...

## Exporting notebook to script

## Reproducibility

## Deployment

## Dependency and environment management

## Containerization

### Building the image

### Running the container

## Cloud deployment

There is no cloud deployment for this project.

## Showcase

```shell
rm -rf content
mkdir -p content/drive/mydrive/mlzoomcamp/capstone1/animals
unzip -o raw-img-20250111T215411Z-001.zip -d content/drive/mydrive/mlzoomcamp/capstone1/animals
```

```shell
/opt/anaconda3/envs/ml-zoomcamp/bin/python -V
# Python 3.9.20
/opt/anaconda3/envs/ml-zoomcamp/bin/python -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.17.1 keras==3.5.0 scipy matplotlib
```

```shell
python prep.py
# ...
python train.py
# ...
python test_model.py
# ...
```
