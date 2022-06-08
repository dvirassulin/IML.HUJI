from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x:  (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y_ = response(X)
    y = y_ + np.random.normal(scale=noise, size=len(y_))

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), y, 2/3)
    train_x, train_y, test_x, test_y = train_x.to_numpy().flatten(), train_y.to_numpy().flatten(), \
                                       test_x.to_numpy().flatten(), test_y.to_numpy().flatten()


    fig = go.Figure(data=[
            go.Scatter(x=X, y=y_, mode="markers+lines", name="Real Points",  marker=dict(color="black", opacity=.7)),
            go.Scatter(x=train_x, y=train_y, mode="markers", name="Predicted train",  marker=dict(color="red", opacity=.7)),
            go.Scatter(x=test_x, y=test_y, mode="markers", name="Predicted test",  marker=dict(color="blue", opacity=.7))],
        layout=go.Layout(title_text=rf"$\text{{Polynomial Fitting of Degree 5 - Sample Noise }}\mathcal{{N}}\left(0,{noise}\right)$",
                         xaxis={"title": r"$x$"},
                         yaxis={"title": r"$y$"}))
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = 10
    train_errors = np.zeros(degrees + 1)
    validate_errors = np.zeros(degrees + 1)
    for k in range(degrees + 1):
        train_errors[k], validate_errors[k] = cross_validate(PolynomialFitting(k), train_x, train_y, mean_square_error, 5)

    x_axis = [k for k in range(degrees + 1)]
    fig1 = go.Figure(data=[
            go.Scatter(x=x_axis, y=train_errors, mode="markers", name="Train error",  marker=dict(color="red", opacity=.7)),
            go.Scatter(x=x_axis, y=validate_errors, mode="markers", name="Validate error",  marker=dict(color="blue", opacity=.7))],
        layout=go.Layout(title_text=rf"$\text{{Errors with cross-validation as function of polynomial degree}}$",
                         xaxis={"title": r"$Degee$"},
                         yaxis={"title": r"$Error$"}))
    fig1.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error

    min_validate_error_degree = np.argmin(validate_errors)
    poly_fit_model = PolynomialFitting(min_validate_error_degree).fit(train_x, train_y)
    test_error = mean_square_error(test_y, poly_fit_model.predict(test_x))
    print(f"The minimal test error is {round(test_error, 2)}, and achieved with polinomial"
          f" degree of {min_validate_error_degree}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(as_frame=True, return_X_y=True)
    train_X = X.sample(50)
    train_y = y[train_X.index]
    test_X = X.drop(train_X.index)
    test_y = y[test_X.index]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = np.linspace(0, 3, n_evaluations)
    ridge_train_err = np.zeros(n_evaluations)
    lasso_train_err = np.zeros(n_evaluations)
    ridge_validate_err = np.zeros(n_evaluations)
    lasso_validate_err = np.zeros(n_evaluations)
    for i in range(n_evaluations):
        lam = lam_range[i]
        ridge_train_err[i], ridge_validate_err[i] = cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error, 5)
        lasso_train_err[i], lasso_validate_err[i] = cross_validate(Lasso(alpha=lam), train_X, train_y, mean_square_error, 5)

    fig1 = go.Figure(data=[
                        go.Scatter(x=lam_range, y=ridge_train_err, mode="lines", name="Ridge train error",  marker=dict(color="red", opacity=.7)),
                        go.Scatter(x=lam_range, y=ridge_validate_err, mode="lines", name="Ridge validate error",  marker=dict(color="blue", opacity=.7)),
                        go.Scatter(x=lam_range, y=lasso_train_err, mode="lines", name="Lasso train error",  marker=dict(color="green", opacity=.7)),
                        go.Scatter(x=lam_range, y=lasso_validate_err, mode="lines", name="Lasso validate error",  marker=dict(color="black", opacity=.7))
                        ],
                    layout = go.Layout(title_text=rf"$\text{{Errors with cross-validation as function regularization parameter}}$",
                    xaxis={"title": r"$Regularization parameter$"},
                    yaxis={"title": r"$Error$"}))
    fig1.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_val_ridge = lam_range[np.argmin(ridge_validate_err)]
    min_val_lasso = lam_range[np.argmin(lasso_validate_err)]

    lasso = Lasso(min_val_lasso).fit(train_X, train_y)
    ridge = RidgeRegression(min_val_ridge).fit(train_X, train_y)
    linear_regression = LinearRegression().fit(train_X, train_y)

    lasso_test_err = mean_square_error(test_y, lasso.predict(test_X))
    ridge_test_err = mean_square_error(test_y, ridge.predict(test_X))
    linear_regression_test_err = mean_square_error(test_y, linear_regression.predict(test_X))

    print(f"The losses for the best regularization parameter are:\n"
          f"lasso: {lasso_test_err} with regularization parameter of {min_val_lasso}\n"
          f"Ridge: {ridge_test_err} with regularization parameter of {min_val_ridge}\n"
          f"Linear regression {linear_regression_test_err}")

if __name__ == '__main__':
    # np.random.seed(0)
    # n_samples, noises = [100, 100, 1500], [5, 0, 10]
    # for sample_size, noise in zip(n_samples, noises):
    #     select_polynomial_degree(sample_size, noise)
    select_regularization_parameter()

