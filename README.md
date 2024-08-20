# Matrix Factorization for Recommendation Systems

## Directory

This repository contains the code for my thesis project. It is organized into the following folders:

1. `Data`: This folder contains the datasets used in this projects.
    
    1.1. `Netflix`: This folder contains the Netflix Prize dataset and the corresponding data extractor. Due to its large size, the full Netflix Prize dataset was not uploaded here.

    1.2. `MovieLens`: This folder contains the MovieLens datasets and the corresponding data extractor. Due to its large size, the MovieLens 25M dataset was not uploaded.

    1.3. `Binary`: This folder contains various binary recommendation system datasets and the corresponding data extractor. 

2. `Library`: This folder contains the modules used in this project.

    2.1. `Metrics`: This folder contains the metrics used for model evaluation, including RMSE and NMSE. They inherit from `tf.keras.metrics.Metric`.

    2.2. `Modules`: This folder contains the matrix factorization algorithms used in this project. They inherit from `tf.Module`.

3. `Experiment`: This folder contains all notebooks in this project. Notably:

    3.1. `movielens25m_df.ipynb` contains the experiments using Dictionary Filter on the Movie Lens 25M dataset.

    3.2. `movielens25m_sgdmf.ipynb` contains the experiments using SGDMF on the Movie Lens 25M dataset.

    3.3. `netflix_df.ipynb` contains the experiments using Dictionary Filter on the Netflix Prize dataset.

    3.4. `netflix_sgdmf.ipynb` contains the experiments using SGDMF on the Netflix Prize dataset.

    3.5. `movielens100k.ipynb` contains the experiments on the Movie Lens 100K dataset.

## Setup

All scripts in this project run on Python `3.10.12`, Tensorflow `2.15.1`. Please refer to `requirements.txt` for the complete list of packages.
