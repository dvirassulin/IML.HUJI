import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df.dropna(inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("C:\\Users\\DvirAssulin\\Desktop\\university\\IML.HUJI\\datasets\\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_Israel = df[df["Country"] == "Israel"]
    df_Israel = df_Israel[df_Israel["Temp"] > -5]
    fig1 = go.Figure(go.Scatter(x=df_Israel["DayOfYear"], y=df_Israel["Temp"], mode='markers', marker=dict(color=df_Israel["Year"]), showlegend=False))
    fig1.update_layout(title_text="$\\text{Temperature as Function of Day of Year}$", title_x = 0.5)
    fig1.update_xaxes(title_text="$\\text{Day of Year}$")
    fig1.update_yaxes(title_text="$Temp$")
    fig1.show()
    monthly_temp_Israel = df_Israel.groupby("Month").agg("std")
    fig2 = go.Figure(go.Bar(x=monthly_temp_Israel.index, y=monthly_temp_Israel["Temp"], marker=dict(color="blue"), showlegend=False))
    fig2.update_layout(title_text="$\\text{Standard Deviation of Temperature by Month}$", title_x = 0.5)
    fig2.update_xaxes(title_text="$\\text{Month}$")
    fig2.update_yaxes(title_text="$Stdev$")
    fig2.show()

    # Question 3 - Exploring differences between countries
    monthly_temp_std = df.groupby("Month").agg("std")
    monthly_temp_avg = df.groupby("Month").agg("mean")
    month_country_temp_std = df.groupby(["Country", "Month"]).agg("std")
    month_country_temp_avg = df.groupby(["Country", "Month"]).agg("mean")
    month_country_temp_avg.reset_index(inplace=True)
    fig3 = px.line(month_country_temp_avg,"Month", "Temp", color="Country", error_y=month_country_temp_std["Temp"])
    fig3.add_trace(go.Scatter(x=monthly_temp_avg.index,y=monthly_temp_avg["Temp"],name="All Countries"))
    fig3.update_layout(title_text="$\\text{Average Temperature of All Countries by Month}$", title_x = 0.5)
    fig3.update_xaxes(title_text="$\\text{Month}$")
    fig3.update_yaxes(title_text="$Mean$")
    fig3.show()


    # Question 4 - Fitting model for different values of `k`
    y = df_Israel["Temp"]
    X = df_Israel["DayOfYear"]
    train_x = X.sample(frac=0.75)
    train_y = y.loc[train_x.index]
    X.drop(train_x.index, inplace=True)
    y.drop(train_y.index, inplace=True)
    test_x = X
    test_y = y
    loss_pred = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(train_x.to_numpy(), train_y.to_numpy())
        loss = poly_model.loss(test_x.to_numpy(), test_y.to_numpy())
        loss = loss.round(2)
        loss_pred.append(loss)
        print(f"The loss for degree of {k} is {loss}\n")
    fig4 = go.Figure(go.Bar(x=[k for k in range(1, 11)], y=loss_pred, marker=dict(color="green"), showlegend=False))
    fig4.update_layout(title_text="$\\text{Loss Per Degree of K}$", title_x = 0.5)
    fig4.update_xaxes(title_text="$\\text{Degree}$")
    fig4.update_yaxes(title_text="$Loss$")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    poly_model = PolynomialFitting(5)
    poly_model.fit(df_Israel["DayOfYear"], df_Israel["Temp"])
    countries = ["The Netherlands", "South Africa", "Jordan"]
    country_loss = []
    for country in countries:
        sub_df = df[df["Country"] == country]
        X = sub_df["DayOfYear"]
        y = sub_df["Temp"]
        loss = poly_model.loss(X, y)
        loss.round(2)
        country_loss.append(loss)

    fig5 = go.Figure(go.Bar(x=countries, y=country_loss, marker=dict(color="red"), showlegend=False))
    fig5.update_layout(title_text="$\\text{Loss Per Country}$", title_x=0.5)
    fig5.update_xaxes(title_text="$\\text{Country}$")
    fig5.update_yaxes(title_text="$Loss$")
    fig5.show()