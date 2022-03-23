from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    sample_size = 1000
    samples = np.random.normal(mu, sigma**2, (sample_size, ))
    estimator = UnivariateGaussian()
    estimator.fit(samples)
    print((float_formatter(estimator.mu_), float_formatter(estimator.var_)))

    # Question 2 - Empirically showing sample mean is consistent
    samples_mean = []
    s, e, j = 10, 1000, 10
    for i in range(s, e + 1, j):
        estimator.fit(samples[:i])
        samples_mean.append(estimator.mu_)
    x_axis = np.linspace(10, 1000, 100)
    y_axis = abs(np.array(samples_mean) - mu)
    go.Figure(go.Scatter(x=x_axis, y=y_axis, mode='lines', name=r'$Distance$'),
              layout=go.Layout(title=r"$\text{(5) Distance From Actual Mean As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\\text{Distance from mean}$",
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    samples_pdf = estimator.pdf(samples)
    x_axis = samples
    go.Figure(go.Scatter(x=x_axis, y=samples_pdf, mode='markers', name=r'PDF'),
              layout=go.Layout(title=r"$\text{(5) Samples PDF Under Fitted Model}$",
                               xaxis_title="$m\\text{ - Sample num}$",
                               yaxis_title="r$\\text{PDF}$",
                               height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1,   0.2, 0, 0.5],
                      [0.2, 2,   0, 0],
                      [0,   0,   1, 0],
                      [0.5, 0,   0, 1]])
    sample_size = 1000
    samples = np.random.multivariate_normal(mu, sigma, (sample_size, ))
    estimator = MultivariateGaussian()
    estimator.fit(samples)
    print(estimator.mu_)
    print(estimator.cov_)

    # # Question 5 - Likelihood evaluation
    estimator.cov_ = sigma
    lin_space1 = np.linspace(-10, 10, 200)
    lin_space2 = np.linspace(-10, 10, 200)
    log_likelihoods = []
    for f1 in lin_space1:
        f1_log_likelihoods = []
        for f3 in lin_space2:
            mu = np.array([f1, 0, f3, 0])
            f1_log_likelihoods.append(estimator.log_likelihood(mu, sigma, samples))
        log_likelihoods.append(f1_log_likelihoods)
    ll_np = np.array(log_likelihoods)
    fig = px.imshow(ll_np,
                    labels=dict(x="f1", y="f3", color="Log Likelihood"),
                    x=lin_space1,
                    y=lin_space2)
    fig.show()

    # Question 6 - Maximum likelihood
    max_val = np.max(ll_np)
    max_indices = np.unravel_index(np.argmax(ll_np), ll_np.shape)
    max_f1 = lin_space1[max_indices[0]]
    max_f2 = lin_space2[max_indices[1]]
    print(max_val.round(3), max_f1.round(3), max_f2.round(3))


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()

