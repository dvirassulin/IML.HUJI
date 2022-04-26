from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.sort(np.unique(y))
        self.mu_ = np.zeros([self.classes_.shape[0], X.shape[1]])
        self.vars_ = np.zeros([self.classes_.shape[0], X.shape[1]])
        self.pi_ = np.zeros(self.classes_.shape[0])
        for i, k in enumerate(self.classes_):
            indexes = np.where(y == k)
            samples = X[indexes]
            self.pi_[i] = len(indexes)/X.shape[0]
            self.mu_[i] = np.mean(samples, axis=0)
            self.vars_[i] = np.var(samples, axis=0)

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
            posterior_mat[:, i] = log_pi_vec + np.sum((-0.5*np.log(self.vars_[i]))-(X-self.mu_[i])**2/(2*self.vars_[i]), axis=1)
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
