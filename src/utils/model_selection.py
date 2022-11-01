from typing import Any, Callable, Dict, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from sklearn.model_selection import GroupKFold

from utils.data import split_prepare

DEFAULT_METRICS = {
    "Accuracy": sklearn.metrics.accuracy_score,
    "F1-Score": sklearn.metrics.f1_score,
    "Recall": sklearn.metrics.recall_score,
    "Precision": sklearn.metrics.precision_score,
}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[
        str,
        Callable[[np.ndarray, np.ndarray], float],
    ] = DEFAULT_METRICS,
) -> Dict[str, float]:
    """Evaluates predictions on given metrics.

    Parameters
    ----------
    y_true: ndarray of shape (n_observations, )
        True outcome values.
    y_pred: ndarray of shape (n_observations, )
        Predicted outcome values.
    metrics: dict of str to callable
        Mapping of metric names to metrics
        (aka. functions taking the true and predicted
        outcomes and returning a float)

    Returns
    -------
    dict of str to float
        Mapping of metric names to scores.
    """
    return {
        metric_name: metric(y_true, y_pred)
        for metric_name, metric in metrics.items()
    }


def evaluate_models(
    models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    ids: Iterable[int],
    n_folds: int = 5,
    n_shuffles: int = 5,
    metrics: Dict[
        str,
        Callable[[np.ndarray, np.ndarray], float],
    ] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Evaluates the given classifiers with repeated k-fold cross validation
    on a set of metrics.

    Parameters
    ----------
    models: dict
        Mapping of model names to models.
    X: ndarray of shape (n_observations, n_features)
        Feature matrix.
    y: ndarray of shape (n_observations, )
        Outcome class labels.
    ids: iterable of int
        Participant ids. Length has to be n_observations
    n_folds: int, default 5
        Number of folds.
    n_shuffles: int, default 5
        Number of times to repeat the k-fold cross validation with a shuffled
        dataset.
    metrics: dict of str to callable
        Mapping of metric names to metrics
        (aka. functions taking the true and predicted
        outcomes and returning a float)

    Returns
    -------
    DataFrame
        Table containing information about classification performance.
    """
    records = []
    n_observations = X.shape[0]
    # Asserting that X and y have the same number of observations
    participant_id = np.array(ids)
    assert participant_id.shape[0] == n_observations
    assert X.shape[0] == y.shape[0]
    for _i_shuffle in range(n_shuffles):
        # Shuffling dataset
        shuffle_indices = np.random.permutation(n_observations)
        X, y, participant_id = (
            X[shuffle_indices],
            y[shuffle_indices],
            participant_id[shuffle_indices],
        )
        # Setting up Kfold cross validation
        kfold = GroupKFold(n_splits=n_folds)
        for train_index, test_index in kfold.split(
            X, y, groups=participant_id
        ):
            # Getting training and testing data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for model_name, model in models.items():
                # Fit each model on a fold
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                record: Dict[str, Any] = dict(model=model_name)
                # Calculate each metric
                for metric_name, metric in metrics.items():
                    score = metric(y_test, y_pred)
                    record[metric_name] = score
                records.append(record)
    return pd.DataFrame.from_records(records)


def select_model(
    evaluation: pd.DataFrame,
    metric: str = "Accuracy",
    ascending: bool = True,
    aggregation: Callable = np.median,
) -> str:
    """Selects model based on the provided evaluation results.

    Parameters
    ----------
    evaluation: DataFrame
        Results of the evaluation.
    metric: str, default 'Accuracy'
        Metric, based on which the model should be selected.
    ascending: bool, default True
        Indicates whether higher=better.
    aggregation: callable, default np.median
        Statistic to aggregate over the different folds and shuffles.

    Returns
    -------
    str
        Name of the selected model.
    """
    summary = evaluation.groupby("model").agg(metric=(metric, aggregation))
    summary = summary.sort_values(by="metric", ascending=ascending)
    best = summary.tail(n=1)
    (best_model,) = best.index
    return best_model


class SelectionProcess:
    """Utility class for automatising the model selection process
    on the dataset.

    Parameters
    ----------
    models: dict of str to callable
        Mapping of model names to functions which create the models from a list
        of participant IDs. This is necessary for hierarchical models.
    data: DataFrame
        Dataset.
    holdout: float, default 0.1
        Indicates how much of the dataset should be left as holdout.
    n_folds: int, default 5
        Number of folds.
    n_shuffles: int, default 5
        Number of times to repeat the k-fold cross validation with a shuffled
        dataset.
    metric: str, default 'Accuracy'
        Metric, based on which the model should be selected.
    metric_ascending: bool, default True
        Indicates whether higher=better.
    metric_aggregation: callable, default np.median
        Statistic to aggregate over the different folds and shuffles.
    """

    def __init__(
        self,
        models: Dict[str, Callable[[Iterable[int]], Any]],
        data: pd.DataFrame,
        holdout: float = 0.1,
        n_folds: int = 5,
        n_shuffles: int = 5,
        metric: str = "Accuracy",
        metric_ascending: bool = True,
        metric_aggregation=np.median,
    ):
        self.models = models
        self.data = data
        self.metric = metric
        self.metric_ascending = metric_ascending
        self.metric_aggregation = metric_aggregation
        self.n_folds = n_folds
        self.n_shuffles = n_shuffles
        (
            self.X_train,
            self.y_train,
            self.X_holdout,
            self.y_holdout,
        ) = split_prepare(data, include_id=True, split=1 - holdout)
        self.ids = self.X_train[:, -1]
        self.X_train = self.X_train[:, :-1]
        self.X_holdout = self.X_holdout[:, :-1]
        self.best_model = None
        self.best_name = None
        self.evaluation = None

    def fit(self):
        """Fitting the selection process, aka running the evaluation,
        selecting and fitting the best performing model on the training data
        by performing repeated k-fold cross validation.
        """
        self.evaluation = evaluate_models(
            self.models,
            self.X_train,
            self.y_train,
            self.ids,
            n_folds=self.n_folds,
            n_shuffles=self.n_shuffles,
        )
        self.best_name = select_model(
            self.evaluation,
            metric=self.metric,
            ascending=self.metric_ascending,
            aggregation=self.metric_aggregation,
        )
        self.best_model = self.models[self.best_name]
        self.best_model.fit(self.X_train, self.y_train)

    def evaluate(self) -> Dict[str, float]:
        """Evaluates selected model on holdout data.

        Returns
        -------
        dict of str to float
            Performance of the selected classifier on multiple metrics.
        """
        assert self.best_model is not None
        y_pred = self.best_model.predict(self.X_holdout)
        return evaluate(self.y_holdout, y_pred)

    def plot_comparison(self) -> go.Figure:
        """Plots comparison between the classifiers on all metrics.

        Returns
        -------
        Figure
            Faceted boxplot of performance comparisons.
        """
        assert self.evaluation is not None
        evaluation_long = self.evaluation.melt(
            id_vars=["model"],
            value_vars=self.evaluation.columns[1:],
            value_name="score",
            var_name="metric",
        )
        fig = px.box(
            evaluation_long,
            x="model",
            y="score",
            facet_col="metric",
            facet_col_wrap=2,
            color="model",
        )
        return fig
