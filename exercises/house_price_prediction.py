from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # df = pd.read_csv(filename, index_col=0)
    # df.reset_index(drop=True, inplace=True)
    # df = df[df['price'] >= 0]
    # df = df[df['sqft_living'] >= 0]
    # df = df[df['sqft_lot'] >= 0]
    # df = df[df['sqft_above'] >= 0]
    # df = df[df['sqft_basement'] >= 0]
    # df = df[df['yr_built'] >= 0]
    # df = df[(df['yr_built'] <= df['yr_renovated']) | (df['yr_renovated'] == 0)]
    # df['built_new'] = df['yr_built'].apply(lambda x: 1 if x >= 1980 else 0)
    # df['renovated_new'] = df['yr_renovated'].apply(lambda x: 1 if x >= 1980 else 0)
    # df.drop(['date', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_lot15', 'sqft_living15'], axis=1, inplace=True)
    # df.dropna(inplace=True)
    #
    # zip_data_frame = pd.get_dummies(df['zipcode'])
    # zip_data_frame.drop(zip_data_frame.columns[0], axis=1, inplace=True)
    # zip_data_frame = zip_data_frame.add_prefix("zip_")
    # df = pd.concat([df, zip_data_frame], axis=1)
    # df.drop(['zipcode'], axis=1, inplace=True)
    # return df
    df = pd.read_csv(filename, index_col=0)
    df.reset_index(drop=True, inplace=True)
    df = df[df['price'] >= 0]
    df = df[df['sqft_living'] >= 0]
    df = df[df['sqft_lot'] >= 0]
    df = df[df['sqft_above'] >= 0]
    df = df[df['sqft_basement'] >= 0]
    df = df[df['yr_built'] >= 0]
    df = df[(df['yr_built'] <= df['yr_renovated']) | (df['yr_renovated'] == 0)]
    df['new_building'] = df['yr_built'].apply(lambda x: 1 if x >= 1990 else 0)
    df['recently_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x >= 1990 else 0)
    df.drop(['date', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_lot15', 'sqft_living15'], axis=1, inplace=True)
    df.dropna(inplace=True)
    zip_data_frame = pd.get_dummies(df['zipcode'])
    zip_data_frame.drop(zip_data_frame.columns[0], axis=1, inplace=True)
    zip_data_frame = zip_data_frame.add_prefix("zip_")
    df = pd.concat([df, zip_data_frame], axis=1)
    df.drop(['zipcode'], axis=1, inplace=True)
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_y = np.std(y)
    for column in X:
        pearson_corr = ((X[column] * y).mean() - X[column].mean() * y.mean()) / (X[column].std() * std_y)
        fig = go.Figure(go.Scatter(x=X[column], y=y, mode='markers', marker=dict(color="black"), showlegend=False))
        fig.update_layout(
            title_text=f"$\\text{{Pearson Correlation of {X[column].name.title()} Feature and Response is {pearson_corr}}}$",
            title_x=0.5)
        fig.update_xaxes(title_text=f"{X[column].name.title()}")
        fig.update_yaxes(title_text="Response")
        path = f"{output_path}\\{X[column].name.title()}_feature.png"
        fig.write_image(path)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    processed_df = load_data("C:\\Users\\DvirAssulin\\Desktop\\university\\IML.HUJI\\datasets\\house_prices.csv")
    X = processed_df.iloc[:, 1:]
    y = processed_df["price"]

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:\\Users\\DvirAssulin\\Desktop\\university\\IML.HUJI\\exercises")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression_model = LinearRegression(True)
    mean_pred = []
    var_pred = []
    x = [i for i in range(10, 101)]
    for p in x:
        pred_loss = []
        for i in range(10):
            data_to_fit = train_x.sample(frac=p / 100)
            response = train_y.loc[data_to_fit.index]
            linear_regression_model.fit(data_to_fit.to_numpy(), response.to_numpy())
            loss = linear_regression_model.loss(np.c_[np.ones(test_x.shape[0]), test_x.to_numpy()], test_y.to_numpy())
            pred_loss.append(loss)
        mean_pred.append(np.mean(pred_loss))
        var_pred.append(np.var(pred_loss))
    fig = go.Figure(data=[
        go.Scatter(x=x, y=mean_pred, mode="markers+lines", name="Mean Prediction", line=dict(dash="dash"),
                   marker=dict(color="green", opacity=.7)),
        go.Scatter(x=x, y=np.array(mean_pred) - 2 * np.sqrt(var_pred), fill=None, mode="lines",
                   line=dict(color="lightgrey"), showlegend=False),
        go.Scatter(x=x, y=np.array(mean_pred) + 2 * np.sqrt(var_pred), fill='tonexty', mode="lines",
                   line=dict(color="lightgrey"), showlegend=False)],
                    layout=go.Layout(title_text="$\\text{Mean Loss by Percentage of Train}$",
                                     xaxis={"title": "$\\text{Percentage of Train Set}$"},
                                     yaxis={"title": "$\\text{Mean Loss}$", "range": [-6, 10]}))
    fig.update_layout(title_x=0.5)
    fig.show()