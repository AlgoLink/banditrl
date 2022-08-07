"""Contextual Logistic Bandit Algorithms."""
from dataclasses import dataclass
from typing import Optional

from sklearn.utils import check_scalar
from sklearn.utils import check_random_state
from scipy.optimize import minimize

import numpy as np

from banditrl.bandit.base import BaseBandit
from .utils import check_array,sigmoid


@dataclass
class LogisticUCB(BaseBandit):
    """Logistic Upper Confidence Bound.
    Parameters
    ------------
    dim: int
        Number of dimensions of context vectors.
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.
    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    alpha_: float, default=1.
        Prior parameter for the online logistic regression.
    lambda_: float, default=1.
        Regularization hyperparameter for the online logistic regression.
    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.
    References
    ----------
    Lihong Li, Wei Chu, John Langford, and Robert E Schapire.
    "A Contextual-bandit Approach to Personalized News Article Recommendation," 2010.
    """
    def __init__(self,
                 history_storage,
                 model_storage,
                 action_storage,
                 recommendation_cls=None,
                 n_actions:int = None,
                 random_state=2022,
                 dim:int = 3,
                 epsilon:float =0.2,
                 alpha_: float = 1.0,
                 lambda_: float = 1.0,
                 model_id: str =None):
        super(LogisticUCB, self).__init__(history_storage, 
                                          model_storage,
                                          action_storage,
                                          recommendation_cls)
        """Initialize class."""
        self.epsilon= epsilon
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.alpha_= alpha_
        self.lambda_ = lambda_
        self.n_actions = n_actions
        self.dim = dim
        self.random_state = random_state
        self.random_ = check_random_state(self.random_state)

        check_scalar(self.alpha_, "alpha_", float)
        if self.alpha_ <= 0.0:
            raise ValueError(f"`alpha_`= {self.alpha_}, must be > 0.0.")
        check_scalar(self.lambda_, "lambda_", float)
        if self.lambda_ <= 0.0:
            raise ValueError(f"`lambda_`= {self.lambda_}, must be > 0.0.")
            
        model = self._model_storage.get_model(model_id)
        if model is None:
            model = {}
            alpha_list = self.alpha_ * np.ones(self.n_actions)
            lambda_list = self.lambda_ * np.ones(self.n_actions)
            reward_lists = [[] for _ in np.arange(self.n_actions)]
            model_list = [
                MiniBatchLogisticRegression(
                    lambda_=lambda_list[i],
                    alpha=alpha_list[i],
                    dim=self.dim,
                    random_state=self.random_state,
                )
                for i in np.arange(self.n_actions)
            ]
            #model["alpha_list"] = alpha_list
            #model["lambda_list"] = lambda_list
            model["model_list"] = model_list
            self._model_storage.save_model(model,model_id)

        self.policy_name = f"logistic_ucb_{self.epsilon}"


    def get_action(self, 
                   context: np.ndarray,
                   len_list,
                   request_id=None, 
                   model_id= None) -> np.ndarray:
        """Select action for new data.
        Parameters
        ------------
        context: array-like, shape (1, dim_context)
            Observed context vector.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        if model_id is None:
            model_id = self._model_id
        if self._action_storage.count() == 0:
            return self._get_action_with_empty_action_storage(context,
                                                              len_list)
        model_storage = self._model_storage.get_model(model_id)
        model_list = model_storage.get("model_list")
        theta = np.array(
            [model.predict_proba(context) for model in model_list]
        ).flatten()
        std = np.array(
            [
                np.sqrt(np.sum((model._q ** (-1)) * (context**2)))
                for model in model_list
            ]
        ).flatten()
        ucb_score = theta + self.epsilon * std
        
        recommendation_ids = ucb_score.argsort()[::-1][: len_list].tolist()
        
        recommendations = []  # pylint: disable=redefined-variable-type
        for action_id in recommendation_ids:
            recommendations.append(self._recommendation_cls(
                action=self._action_storage.get(action_id),
                estimated_reward=1.0,
                uncertainty=1.0,
                score=1.0,
            ))
        history_id = self._history_storage.add_history(context, 
                                                       recommendations,
                                                       request_id=request_id, 
                                                       model_id=model_id)
            
        return recommendations

    def reward(self, history_id, action: int, reward: float,model_id=None) -> None:
        """Update policy parameters.
        Parameters
        ----------
        action: int
            Selected action by the policy.
        reward: float
            Observed reward for the chosen action and position.
        context: array-like, shape (1, dim_context)
            Observed context vector.
        """
        if model_id is None:
            model_id = self._model_id

        context = (self._history_storage
                   .get_unrewarded_history(history_id,model_id= model_id)
                   .context)

        # Update the model
        model = self._model_storage.get_model(model_id)
        model_list =  model["model_list"]
        _model = model_list[int(action)]
        _model.fit(
            X=np.concatenate([context], axis=0),
            y=np.array([reward]),
        )
        model_list[int(action)]=_model
        model["model_list"] = model_list
        self._model_storage.save_model(model,model_id)

        return True
@dataclass
class MiniBatchLogisticRegression:
    """MiniBatch Online Logistic Regression Model."""

    def __init__(self,lambda_: float,
                 alpha: float,
                 dim: int,
                 random_state: Optional[int] = None) -> None:
        """Initialize Class."""
        self.dim = dim
        self.lambda_ = lambda_
        self.random_state = random_state
        self.alpha = alpha

        self._m = np.zeros(self.dim)
        self._q = np.ones(self.dim) * self.lambda_
        self.random_ = check_random_state(self.random_state)

    def loss(self, w: np.ndarray, *args) -> float:
        """Calculate loss function."""
        X, y = args
        return (
            0.5 * (self._q * (w - self._m)).dot(w - self._m)
            + np.log(1 + np.exp(-y * w.dot(X.T))).sum()
        )

    def grad(self, w: np.ndarray, *args) -> np.ndarray:
        """Calculate gradient."""
        X, y = args
        return self._q * (w - self._m) + (-1) * (
            ((y * X.T) / (1.0 + np.exp(y * w.dot(X.T)))).T
        ).sum(axis=0)

    def sample(self) -> np.ndarray:
        """Sample coefficient vector from the prior distribution."""
        return self.random_.normal(self._m, self.sd(), size=self.dim)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Update coefficient vector by the mini-batch data."""
        self._m = minimize(
            self.loss,
            self._m,
            args=(X, y),
            jac=self.grad,
            method="L-BFGS-B",
            options={"maxiter": 20, "disp": False},
        ).x
        P = (1 + np.exp(1 + X.dot(self._m))) ** (-1)
        self._q = self._q + (P * (1 - P)).dot(X**2)

    def sd(self) -> np.ndarray:
        """Standard deviation for the coefficient vector."""
        return self.alpha * (self._q) ** (-1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the expected coefficient."""
        return sigmoid(X.dot(self._m))

    def predict_proba_with_sampling(self, X: np.ndarray) -> np.ndarray:
        """Predict extected probability by the sampled coefficient."""
        return sigmoid(X.dot(self.sample()))