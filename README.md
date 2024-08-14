# Matrix Factorization for Recommendation Systems

## Directory

This repository contains the code for my thesis project. It is organized into the following folders:

1. `Data`: This folder contains the datasets used in this projects.
    
    1.1. `Netflix`: This folder contains the Netflix Prize dataset and the corresponding data extractor. The full Netflix Prize dataset was not uploaded.

    1.2. `MovieLens`: This folder contains the MovieLens datasets and the corresponding data extractor. The MovieLens 25M dataset was not uploaded.

    1.3. `Binary`: This folder contains various binary recommendation system datasets and the corresponding data extractor. 

2. `Library`: This folder contains the modules used in this project.

    2.1. `Metrics`: This folder contains the metrics used for model evaluation, including RMSE and NMSE. They inherit from `tf.keras.metrics.Metric`.

    2.2. `Modules`: This folder contains the matrix factorization algorithms used in this project. They inherit from `tf.Module`.

3. `Experiment`: This folder contains all notebooks in this project.

## Setup

All scripts in this project run on Python `3.10.12`, Tensorflow `2.15.1`. Checkout `requirements.txt` for the complete list of packages.
