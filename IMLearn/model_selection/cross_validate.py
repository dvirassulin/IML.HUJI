from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    losses_validation = np.zeros(cv)
    losses_train = np.zeros(cv)
    folds_X = np.array_split(X, cv)
    folds_y = np.array_split(X, cv)
    for i in range(cv):
        folds_X_no_i = folds_X[:i] + folds_X[i+1:]
        folds_y_no_i = folds_y[:i] + folds_y[i+1:]
        estimator.fit(np.concatenate(folds_X_no_i), np.concatenate(folds_y_no_i))
        losses_validation[i] = scoring(estimator.predict(folds_X[i]), folds_y[i])
        losses_train[i] = scoring(estimator.predict(folds_X[i]), folds_y[i])
    return np.average(losses_train), np.average(losses_validation)


