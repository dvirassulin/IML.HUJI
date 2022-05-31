import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
from IMLearn.metrics.loss_functions import accuracy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from os.path import exists
import json


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    train_error = np.array([adaboost.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)])
    test_error = np.array([adaboost.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)])
    x_axis = [i for i in range(1, n_learners + 1)]

    fig1 = go.Figure([go.Scatter(x=x_axis, y=train_error, name="Train error", showlegend=True,
                                marker=dict(color="blue", opacity=.7),
                                line=dict(color="blue", width=2))],
                    layout=go.Layout(title=rf"$\textbf{{(1) Loss as function of number of learners with noise {noise}}}$",
                                     xaxis={"title": "number of learners"},
                                     yaxis={"title": "loss"},
                                     height=400))
    fig1.add_trace(go.Scatter(x=x_axis, y=test_error, name="Test error", showlegend=True,
                                marker=dict(color="orange", opacity=.7),
                                line=dict(color="orange", width=2)))
    fig1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise == 0:
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} learners}}$" for t in T],
                            horizontal_spacing=0.05, vertical_spacing=.15)
        for i, t in enumerate(T):
            fig2.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                            go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                       marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=0.1)))],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)
        fig2.update_layout(title=rf"$\textbf{{(2) Decision Boundaries With Different Learners Num - {noise} Noise}}$", margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig2.show()

    # Question 3: Decision surface of best performing ensemble
    losses = np.array([adaboost.partial_loss(test_X, test_y, t) for t in range(1, adaboost.iterations_ + 1)])
    minimal_loss_ensemble_size = np.argmin(losses) + 1
    if noise == 0:
        fig3 = make_subplots(rows=1, cols=1, subplot_titles=[rf"$\textbf{{{minimal_loss_ensemble_size} learners}}$"],
                            horizontal_spacing=0.05, vertical_spacing=.15)
        fig3.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, minimal_loss_ensemble_size), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=1, cols=1)
        adaboost_accuracy = accuracy(test_y, adaboost.partial_predict(test_X, minimal_loss_ensemble_size))
        fig3.update_layout(title=rf"$\textbf{{(3) Decision Boundaries With Minimal Loss - {adaboost_accuracy} Accuracy}}$",
                           margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig3.show()

    # Question 4: Decision surface with weighted samples
    fig4 = make_subplots(rows=1, cols=1, subplot_titles=[rf"$\textbf{{{adaboost.iterations_} learners}}$"],
                         horizontal_spacing=0.05, vertical_spacing=.15)
    fig4.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1),
                                            size=(adaboost.D_ / np.max(adaboost.D_)) * 5))],
                    rows=1, cols=1)
    fig4.update_layout(title=rf"$\textbf{{(4) Decision Boundaries With Weighted Samples - {noise} Noise}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig4.show()


def serialize_and_save_model(adaboost, json_name):
    json_dict = {"models":
                     [{"j": m.j_, "threshold": m.threshold_, "sign": m.sign_} for m in adaboost.models_],
                 "iterations": adaboost.iterations_,
                 "D": list(adaboost.D_),
                 "weights": list(adaboost.weights_)
                 }
    json_data = json.dumps(json_dict, indent=4)
    with open(json_name, 'w') as outfile:
        outfile.write(json_data)


def deserialize_model(json_name):
    with open(json_name) as json_file:
        adaboost_dict = json.load(json_file)
        adaboost = AdaBoost(DecisionStump, adaboost_dict["iterations"])
        adaboost.iterations_ = adaboost_dict["iterations"]
        models_list_dicts = np.array(adaboost_dict["models"])
        adaboost.models_ = [DecisionStump() for i in range(adaboost.iterations_)]
        for i in range(adaboost.iterations_):
            adaboost.models_[i].j_ = models_list_dicts[i]["j"]
            adaboost.models_[i].sign_ = models_list_dicts[i]["sign"]
            adaboost.models_[i].threshold_ = models_list_dicts[i]["threshold"]
            adaboost.models_[i].fitted_ = True
        adaboost.D_ = np.array(adaboost_dict["D"])
        adaboost.weights_ = np.array(adaboost_dict["weights"])
        adaboost.fitted_ = True
    return adaboost


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise, n_learners=250)
