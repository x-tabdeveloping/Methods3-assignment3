from typing import Any, Callable, Dict, Optional

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (SelectFromModel, SelectPercentile,
                                       f_classif)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

FEATURE_SELECTORS = {
    "lasso": SelectFromModel(Lasso(alpha=0.1)),
    "logistic_lasso": SelectFromModel(
        LogisticRegression(penalty="l1", solver="saga", C=1.0)
    ),
    "trees": SelectFromModel(ExtraTreesClassifier()),
    "anova": SelectPercentile(score_func=f_classif, percentile=10),
}


def create_pipeline(model, feature_selector: Optional[str]) -> Pipeline:
    """Adds z-score scaler and feature selection to model."""
    pipe = [("scaler", StandardScaler())]
    if feature_selector is not None:
        pipe.append(
            (
                "feature_selection",
                FEATURE_SELECTORS[feature_selector],
            )
        )
    pipe.append(("model", model))
    return Pipeline(pipe)


def wrap(
    models: Dict[str, Any], feature_selector: Optional[str] = None
) -> Dict[str, Pipeline]:
    """Wraps all models in a pipeline with the chosen feature selector"""
    pipelines = {
        model_name: create_pipeline(model, feature_selector)
        for model_name, model in models.items()
    }
    return pipelines
