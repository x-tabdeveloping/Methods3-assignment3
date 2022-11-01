from typing import Tuple

import numpy as np
import pandas as pd

def prepare(
    data: pd.DataFrame, include_id: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Turns the dataset into a design matrix and outcome vector.

    Parameters
    ----------
    data: DataFrame
        Raw dataset.
    include_id: bool, default False
        Indicates whether the set should include participant ids.

    Returns
    -------
    X: ndarray of shape (n_observations, n_features)
        Design matrix.
    y: ndarray of shape (n_observations, )
        Vector of outcomes.
    """
    features = pd.Series(data.columns)
    features = features[~features.isin(["participant_id", "trial", "diagnosis"])]
    features = features[~features.str.endswith("_mean")]
    features = features.tolist()
    if include_id:
        features.append("participant_id")
    X = data[features].to_numpy()
    y = data["diagnosis"].to_numpy()
    return X, y


def train_test_split(
    data: pd.DataFrame, split: float = 0.9
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into test and split data.

    Parameters
    ----------
    data: DataFrame
        Dataset.
    split: float, default 0.9
        Indicates how much of the dataset should be training data.

    Returns
    -------
    training_data: DataFrame
        Training data.
    test_data: DataFrame
        Test data.
    """
    participants = data.participant_id.unique()
    np.random.shuffle(participants)
    n_participants = participants.shape[0]
    n_training = int(n_participants * split)
    training_participants = participants[:n_training]
    test_participants = participants[n_training:]
    training_data = data[data.participant_id.isin(training_participants)]
    test_data = data[data.participant_id.isin(test_participants)]
    return training_data, test_data


def split_prepare(
    data: pd.DataFrame, split: float = 0.9, include_id: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits and prepares data for training.

    Parameters
    ----------
    data: DataFrame
        Dataset.
    split: float, default 0.9
        Indicates how much of the dataset should be training data.
    include_id: bool, default False
        Indicates whether the set should include participant ids.

    Returns
    -------
    X_train: ndarray of shape (n_observations*split, n_features)
    y_train: ndarray of shape (n_observations*split, )
    X_test: ndarray of shape (n_observations*(1-split), n_features)
    y_test: ndarray of shape (n_observations*(1-split), )
    """
    training_data, test_data = train_test_split(data, split=split)
    X_train, y_train = prepare(training_data, include_id=include_id)
    X_test, y_test = prepare(test_data, include_id=include_id)
    return X_train, y_train, X_test, y_test
