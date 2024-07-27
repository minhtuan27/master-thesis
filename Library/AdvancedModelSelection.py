import pandas as pd

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
    grouped = dataset.groupby('User Index', group_keys=False)
    test_set = grouped.apply(lambda x: x.sample(frac=test_size, random_state=random_state))
    train_set = pd.concat([dataset, test_set]).drop_duplicates(keep=False)

    grouped = train_set.groupby('User Index', group_keys=False)
    val_set = grouped.apply(lambda x: x.sample(frac=val_size, random_state=random_state))
    train_set = pd.concat([train_set, val_set]).drop_duplicates(keep=False)

    return train_set, val_set, test_set