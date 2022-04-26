import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        def callback(model, X, y):
            losses.append(model.loss(X, y))

        losses = []
        Perceptron(callback=callback).fit(X, y)
        # Plot figure of loss as function of fitting iteration
        x_axis = np.linspace(0, 1000, 100)
        go.Figure([go.Scatter(x=x_axis, y=losses, mode='lines+markers', name=r'$text{Loss per Iteration}$')],
                  layout=go.Layout(title=f"$\\text{{Loss of Perceptron Algorithm per Iteration in {n} DataSet}}$",
                                   xaxis_title=r"$\text{Iteration Number}$",
                                   yaxis_title=r"$\text{Loss At Iteration}$",
                                   height=500)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """

    symbols = np.array(["circle", "star-diamond", "square"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        # Fit models and predict over training set

        lda_model = LDA()
        naive_bayes_model = GaussianNaiveBayes()
        lda_model.fit(X, y)
        naive_bayes_model.fit(X, y)
        lda_pred = lda_model.predict(X)
        naive_bayes_pred = naive_bayes_model.predict(X)

        models = [GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)]
        model_names = ["Naive bayes", "LDA"]
        predictions = [m.predict(X) for m in models]


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}  Accuracy: {round(accuracy(y, pred), 2)}$"
                                                            for m, pred in zip(model_names, predictions)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, m in enumerate(models):
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=predictions[i], symbol=symbols[y], colorscale=[custom[0], custom[-1]])),
                            go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color="black", symbol="x"))],
                                rows=(i // 3) + 1, cols=(i % 3) + 1)
            if isinstance(m, LDA):
                fig.add_traces([get_ellipse(m.mu_[k], m.cov_) for k in range(m.mu_.shape[0])],
                               rows=(i // 3) + 1, cols=(i % 3) + 1)
            else:
                fig.add_traces([get_ellipse(m.mu_[k], np.diag(m.vars_[k])) for k in range(m.mu_.shape[0])],
                               rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\textbf{{{f} Dataset}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)


        #     raise NotImplementedError()
        #
        # raise NotImplementedError()
        #
        # # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()
        #
        # # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        #
        # # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
