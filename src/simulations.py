# %% Importing packages
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from IPython import get_ipython
from sklearn.linear_model import LogisticRegression

from utils.bayesian_model import HierarchicalLogisticRegression
from utils.data import split_prepare
from utils.model_selection import evaluate
from utils.simulation import simulate_sample


# %%
def visualize_data(data):
    """Function for visualizing data"""
    data_long = data.melt(
        id_vars=["participant_id", "trial", "diagnosis"],
        value_vars=data.columns[9:],
        value_name="measurement",
    )
    fig = px.histogram(
        data_long,
        x="measurement",
        color="diagnosis",
        facet_col="variable",
        facet_col_wrap=3,
        barmode="overlay",
    )
    return fig


# %% Setting up autoreload
try:
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
except Exception:
    print("Could not load Ipython kernel, continuing without autoreload")

# %% Simulating informed data
informed_data = simulate_sample(n_group_size=100, n_trials=10, informed=True)
informed_data.info()
informed_data

# %% Visualizing simulated data
fig = visualize_data(informed_data)
fig.show()
fig.write_image("../plots/informed_data.png")

# %% Preparing data
X_train, y_train, X_test, y_test = split_prepare(
    informed_data, split=0.9, include_id=True
)
ids = X_train[:, -1]
X_train = X_train[:, :-1]
X_test = X_test[:, :-1]

# %% Fitting model
# Setting the hyperpriors' standard deviations to be 10
model = HierarchicalLogisticRegression(
    participant_id=ids, sigma_mu_beta=10, sigma_sigma_beta=10
).fit(X_train, y_train)

# %% Summary
summary = az.summary(model.trace)
print(summary)
summary.to_csv("../results/simulation_summary.csv")


# %% Investigate feature importance
variables = pd.Series(model.trace.posterior.data_vars.keys())
slopes = variables[variables.str.contains("slope_mean")]
az.plot_posterior(model.trace, var_names=slopes)
plt.savefig("../plots/feature_importance.png")


# %% Evaluate model performance
y_pred = model.predict(X_test)
print(evaluate(y_test, y_pred))

# %% Evaluate alternative model
model = LogisticRegression(penalty="l2").fit(X_train, y_train)
y_pred = model.predict(X_test)
print(evaluate(y_test, y_pred))

# SHEEEESH that's some serious code duplication, I don't like this

# %% Simulating uninformed data
uninformed_data = simulate_sample(
    n_group_size=100, n_trials=10, informed=False
)
uninformed_data.info()
uninformed_data

# %% Visualizing simulated data
fig = visualize_data(uninformed_data)
fig.show()
fig.write_image("../plots/uninformed_data.png")

# %% Preparing data
X_train, y_train, X_test, y_test = split_prepare(
    uninformed_data, split=0.9, include_id=True
)
ids = X_train[:, -1]
X_train = X_train[:, :-1]
X_test = X_test[:, :-1]

# %% Fitting model
# Setting the hyperpriors' standard deviations to be 10
model = HierarchicalLogisticRegression(
    participant_id=ids, sigma_mu_beta=10, sigma_sigma_beta=10
).fit(X_train, y_train)

# %% Summary
summary = az.summary(model.trace)
print(summary)
summary.to_csv("../results/uninformed_simulation_summary.csv")


# %% Investigate feature importance
variables = pd.Series(model.trace.posterior.data_vars.keys())
slopes = variables[variables.str.contains("slope_mean")]
az.plot_posterior(model.trace, var_names=slopes)
plt.savefig("../plots/uninformed_feature_importance.png")

# %% Evaluate model performance
y_pred = model.predict(X_test)
evaluate(y_test, y_pred)
