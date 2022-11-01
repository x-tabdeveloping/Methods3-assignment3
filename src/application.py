# %% Importing packages
import numpy as np
import pandas as pd
import plotly.express as px
from IPython import get_ipython
from rvfln.activation import LeakyReLU
from rvfln.rvfln import RVFLNClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from utils.model_selection import SelectionProcess, evaluate
from utils.models import MODELS
from utils.pipeline import wrap

# %% Setting up autoreload
try:
    ipython = get_ipython()
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
except Exception:
    print("Could not load Ipython kernel, continuing without autoreload")

# %% Loading empirical data
data = pd.read_csv("../dat/empirical_data.csv")
print("Before:")
data.info()

# %% Plotting some of the data
fig = px.pie(data, names="Diagnosis")
fig.show()

# %% Cleaning the data
data = data.rename(
    columns={
        "NewID": "participant_id",
        "Diagnosis": "diagnosis",
    }
)
data = data.drop(columns=["Corpus", "Trial", "Language", "Gender", "PatID"])
data["participant_id"], id_uniques = pd.factorize(data["participant_id"])
diagnosis = np.where(data.diagnosis == "SCZ", 1, 0)
data = data.assign(diagnosis=diagnosis)
print("After:")
data.info()

# %% Setting up pipelines
models = {
    "Random Forest": RandomForestClassifier(),
    "L2 Logistic Regression": LogisticRegression(penalty="l2"),
    "RVFLN": RVFLNClassifier(
        n_enhancement=15, alpha=1.0, activation=LeakyReLU
    ),
    "Linear SVM": LinearSVC(),
    "5-nearest Neighbours Classifier": KNeighborsClassifier(n_neighbors=5),
}
models = wrap(models, feature_selector=None)

# %% Setting up selection process
selection = SelectionProcess(
    models=models,
    data=data,
    holdout=0.1,
    n_folds=5,
    n_shuffles=5,
    metric="Accuracy",
    metric_aggregation=np.median,
)

# %% Running model selection
selection.fit()
print("Selected model: ", selection.best_name)

# %% Plotting model comparison
fig = selection.plot_comparison()
fig.update_xaxes(visible=False, showticklabels=False)
fig.show()
fig.write_image("../plots/model_comparison.png")

# %% Evaluating on holdout set
print(selection.evaluate())

# %% Evaluating dummy
dummy = DummyClassifier().fit(selection.X_train, selection.y_train)
y_pred = dummy.predict(selection.X_holdout)
print(evaluate(y_pred, selection.y_holdout))
