from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        shifts = -self.loss_derivative(y, predictions)
        ind = np.random.choice(x.shape[0], size=int(x.shape[0] * self.subsample), replace=True)
        
        model = self.base_model_class(**self.base_model_params)
        model = model.fit(x[ind], shifts[ind])
        gamma = self.find_optimal_gamma(shifts[ind], predictions[ind], model.predict(x[ind]))
        
        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_preds = np.zeros(y_train.shape[0])
        val_preds = np.zeros(y_valid.shape[0])

        self.history['train_loss'].append(self.loss_fn(y_train, train_preds))
        self.history['val_loss'].append(self.loss_fn(y_valid, val_preds))

        rounds = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_preds)

            train_preds += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            val_preds += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)

            self.history['train_loss'].append(self.loss_fn(y_train, train_preds))
            self.history['val_loss'].append(self.loss_fn(y_valid, val_preds))

            if self.early_stopping_rounds is not None:
                if self.history['val_loss'][-2] <= self.history['val_loss'][-1]:
                    rounds = 0
                    continue
                rounds += 1
                if rounds >= self.early_stopping_rounds: 
                    break

        if self.plot:
            plt.plot(np.arange(self.n_estimators), self.history['train_loss'], label='train')
            plt.plot(np.arange(self.n_estimators), self.history['val_loss'], label='validation')

            plt.title('Loss per iteration history')
            plt.xlabel('n_estimators')
            plt.ylabel('loss')
            plt.legend()

            plt.show()

    def predict_proba(self, x):
        preds = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            preds += self.learning_rate * gamma * model.predict(x)
        probs = self.sigmoid(preds).reshape(-1, 1)
        return np.concatenate((1 - probs, probs), axis=1)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        return np.mean(np.array([model.feature_importances_ / sum(model.feature_importances_) for model in self.models]), axis=0)
    