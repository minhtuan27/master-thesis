import pandas as pd
import numpy as np

def user_based_train_val_test_split(dataset: pd.DataFrame, test_size: float, val_size: float, random_state: int = None):
    """
    Group the dataset by user id and split the dataset into training, validation and testing sets.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to split.
    test_size : float
        The proportion of the dataset to include in the test split.
    val_size : float
        The proportion of the train dataset to include in the validation split.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The training set.
    pd.DataFrame
        The testing set.
    """
    if random_state:
        np.random.seed(random_state)
    
    grouped = dataset.groupby('User Index', group_keys=False)
    
    train_index = []
    val_index = []
    test_index = []

    for _, group in grouped:
        n = len(group)
        test_counts = int(n * test_size)
        val_counts = int(n * val_size * (1 - test_size))
        indices = np.random.permutation(n)
        test_index.extend(group.index[indices[:test_counts]])
        val_index.extend(group.index[indices[test_counts:test_counts + val_counts]])
        train_index.extend(group.index[indices[test_counts + val_counts:]])

    train_set = dataset.loc[train_index]
    val_set = dataset.loc[val_index]
    test_set = dataset.loc[test_index]

    return train_set, val_set, test_set