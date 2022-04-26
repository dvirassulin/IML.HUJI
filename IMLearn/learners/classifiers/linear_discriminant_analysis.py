from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.array(np.sort(np.unique(y)), dtype=int)
        self.mu_ = np.zeros([self.classes_.shape[0], X.shape[1]])
        self.cov_ = np.zeros(self.classes_.shape[0])
        self.pi_ = np.zeros(self.classes_.shape[0])
        mu_per_sample = np.zeros([y.shape[0], X.shape[1]])
        for i, k in enumerate(self.classes_):
            indexes = np.where(y == k)
            samples = X[indexes]
            self.pi_[i] = len(indexes)/X.shape[0]
            self.mu_[i, :] = np.mean(samples, axis=0)
            mu_per_sample[indexes] = self.mu_[i]
        self.cov_ = np.sum(np.einsum('ij,ik->ijk', X - mu_per_sample, X - mu_per_sample), axis=0) / (X.shape[0] - len(self.classes_))


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        posterior = self.likelihood(X)
        indexes = np.argmax(posterior, axis=1)
        labels = np.zeros(X.shape[0], dtype=int)
        for i, index in enumerate(indexes):
            labels[i] = self.classes_[index]
        return labels

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        posterior_mat = np.zeros([X.shape[0], len(self.classes_)])
        for i, k in enumerate(self.classes_):
            log_pi_vec = np.full(shape=X.shape[0], fill_value=np.log(self.pi_[i]))
            mu_cov_vec = np.full(shape=X.shape[0], fill_value=0.5*self.mu_[i]@np.linalg.inv(self.cov_)@self.mu_[i])
            posterior_mat[:, i] = log_pi_vec + X@np.linalg.inv(self.cov_)@self.mu_[i] - mu_cov_vec
        return posterior_mat

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
