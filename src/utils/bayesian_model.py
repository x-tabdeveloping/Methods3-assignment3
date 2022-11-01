from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import pymc as pm
from pymc.sampling_jax import sample_blackjax_nuts
from scipy.special import expit


class HierarchicalLogisticRegression:
    """Bayesian hierarchical logistic regression model with population and
    participant level estimates.

    Parameters
    ----------
    participant_id: iterable of int
        IDs of the participants. Has to be length n_observations.
    sigma_mu_beta: float, default 1
        Standard deviation of prior of population level slope mean estimates.
    sigma_sigma_beta: float, default 1
        Standard deviation of prior of population level slope
        standard deviation estimates.

    Note
    ----
    The purpose of this class is to provide compatibility with sklearn's API.
    """

    def __init__(
        self,
        participant_id: Iterable[int],
        sigma_mu_beta: float = 1,
        sigma_sigma_beta: float = 1,
    ):
        self.trace = None
        self.model = None
        self.intercept = 0
        self.slope = np.array(0)
        self.participant_id = participant_id
        self.sigma_mu_beta = sigma_mu_beta
        self.sigma_sigma_beta = sigma_sigma_beta

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> HierarchicalLogisticRegression:
        """Fits the model on the given training data.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Feature matrix.
        y: ndarray of shape (n_observations, )
            Outcome values.

        Returns
        -------
        HierarchicalLogisticRegression
            The fitted model.
        """
        id_values, id_uniques = pd.factorize(self.participant_id)
        coords = {"participants": id_uniques}
        n_features = X.shape[1]
        with pm.Model(coords=coords) as model:
            participant_id = pm.MutableData("participant_id", id_values)
            features: List[pm.MutableData] = []
            mean_beta: List[pm.Normal] = []
            sigma_beta: List[pm.HalfNormal] = []
            beta: List[pm.Normal] = []
            mean_beta_0 = pm.Normal(
                "pop_intercept_mean", mu=0, sigma=self.sigma_mu_beta
            )
            sigma_beta_0 = pm.HalfNormal(
                "pop_intercept_sigma", self.sigma_sigma_beta
            )
            beta_0 = pm.Normal(
                "intercept",
                mu=mean_beta_0,
                sigma=sigma_beta_0,
                dims="participants",
            )
            for i_feature in range(n_features):
                features.append(
                    pm.MutableData(f"feature_[{i_feature}]", X[:, i_feature])
                )
                mean_beta.append(
                    pm.Normal(
                        f"pop_slope_mean_{i_feature}",
                        mu=0,
                        sigma=self.sigma_mu_beta,
                    )
                )
                sigma_beta.append(
                    pm.HalfNormal(
                        f"pop_slope_sigma_{i_feature}", self.sigma_sigma_beta
                    )
                )
                beta.append(
                    pm.Normal(
                        f"slope_{i_feature}",
                        mu=mean_beta[i_feature],
                        sigma=sigma_beta[i_feature],
                        dims="participants",
                    )
                )
            log_odds = beta_0[participant_id]
            for i_feature in range(n_features):
                log_odds += (
                    beta[i_feature][participant_id] * features[i_feature]
                )
            probability = pm.Deterministic(
                "probability", pm.math.sigmoid(log_odds)
            )
            diagnosis = pm.Bernoulli("diagnosis", probability, observed=y)
        self.model = model
        with self.model:
            # self.trace = pm.sample(step=pm.Metropolis())
            self.trace = sample_blackjax_nuts()
            self.trace.extend(pm.sample_prior_predictive())
        self.intercept = self.trace.posterior["pop_intercept_mean"].mean()
        self.slope = np.array(
            [
                self.trace.posterior[f"pop_slope_mean_{i_feature}"].mean()
                for i_feature in range(n_features)
            ]
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probability estimates for the given predictor values.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_observations, )
            Probability estimate for each observation.
        """
        if X.shape[1] > self.slope.shape[0]:
            raise ValueError("Shape mismatch, X has too many features.")
        if self.trace is None:
            raise ValueError(
                "Model is not fitted yet, "
                "please fit before trying to use it for prediction"
            )
        log_odds = self.intercept.to_numpy() + X.dot(self.slope)
        probability = expit(log_odds)
        return probability

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts outcomes for the given predictor values.

        Parameters
        ----------
        X: ndarray of shape (n_observations, n_features)
            Feature matrix.

        Returns
        -------
        ndarray of shape (n_observations, )
            Outcome class labels.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)
