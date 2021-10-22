import os
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import yaml


def create_windows(data: np.ndarray, window_size):
    return data[np.arange(window_size) + np.arange(
        data.shape[0] - window_size).reshape(-1, 1)]

def featurize(test, train, window_size):
    """

    Parameters
    ----------
    test : np.ndarray
        of shape [-1, n_features]
    """
    sc = preprocessing.StandardScaler()
    train = sc.fit_transform(train.reshape(-1, 1))
    test = sc.transform(test.reshape(-1, 1))

    window_train = create_windows(train, window_size).reshape(-1, train.shape[
        1]*window_size)
    window_test = create_windows(test, window_size).reshape(-1, test.shape[
        1]*window_size)

    return window_train, window_test