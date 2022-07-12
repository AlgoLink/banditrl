"""Contextual Linear Bandit Algorithms."""
from dataclasses import dataclass
from sklearn.utils import check_scalar
from sklearn.utils import check_random_state

import logging

import six
import numpy as np

from banditrl.bandit.base import BaseBandit
from .utils import check_array

LOGGER = logging.getLogger(__name__)

@dataclass
class BaseLinPolicy(BaseBandit):
    """Base class for contextual bandit policies using linear regression.
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
    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].
    """
    def __init__(self,
                 history_storage,
                 model_storage,
                 action_storage,
                 recommendation_cls=None,
                 n_actions=None,
                 random_state=2022,
                 dim = 3,
                 model_id=None):
        super(BaseLinPolicy, self).__init__(history_storage, 
                                            model_storage,
                                            action_storage,
                                            recommendation_cls)
        self._model_id=model_id
        self.random_ = check_random_state(random_state)
        self.n_actions = n_actions
        # model init
        theta_hat = np.zeros((dim, n_actions))
        A_inv = np.concatenate(
            [np.identity(dim) for _ in np.arange(n_actions)]
        ).reshape(n_actions, dim, dim)
        b = np.zeros((dim, n_actions))
        model = {}
        model["A_inv"] = A_inv
        model["b"] = b
        model["theta_hat"] = theta_hat

        self._model_storage.save_model(model,model_id)

    def _init_action_model(self) -> None:
        """Initialize class."""
        super().__post_init__()
        self.theta_hat = np.zeros((self.dim, self.n_actions))
        self.A_inv = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b = np.zeros((self.dim, self.n_actions))

        self.A_inv_temp = np.concatenate(
            [np.identity(self.dim) for _ in np.arange(self.n_actions)]
        ).reshape(self.n_actions, self.dim, self.dim)
        self.b_temp = np.zeros((self.dim, self.n_actions))

    def reward(self, history_id, rewards,model_id=None) -> None:
        """Update policy parameters.
        Parameters
        ------------
        history_id: str
            request_id
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
                   .get_unrewarded_history(history_id,model_id=model_id)
                   .context)

        # Update the model
        model = self._model_storage.get_model(model_id)
        A_inv_temp = model["A_inv"]
        b_temp =  model["b"]
        # update the inverse matrix by the Woodbury formula
        A_inv_temp[action] -= (
            A_inv_temp[action]
            @ context.T
            @ context
            @ A_inv_temp[action]
            / (1 + context @ A_inv_temp[action] @ context.T)[0][0]
        )
        b_temp[:, action] += reward * context.flatten()
        theta_temp[:, action]=A_inv_temp[action] @ b_temp[:, action]

        self._model_storage.save_model({
            'A_inv': A_inv_temp,
            'b': b_temp,
            'theta_hat': theta_temp,
        },model_id)

        # Update the history
        self._history_storage.add_reward(history_id, rewards,model_id)

@dataclass
class LinEpsilonGreedy(BaseLinPolicy):
    """Linear Epsilon Greedy.
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
    n_trial: int, default=0
        Current number of trials in a bandit simulation.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    epsilon: float, default=0.
        Exploration hyperparameter that must take value in the range of [0., 1.].
    References
    ------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    def __init__(self,epsilon=0.2) -> None:
        """Initialize class."""
        self.epsilon = epsilon
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        self.policy_name = f"linear_epsilon_greedy_{self.epsilon}"
        super().__init__()

    def get_action(self, context: np.ndarray,len_list,model_id= None) -> np.ndarray:
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
        model = self._model_storage.get_model(model_id)
        A_inv = model['A_inv']  # pylint: disable=invalid-name
        theta = model['theta_hat']


        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        context_dim,n_actions = theta.shape
        if self.random_.rand() > self.epsilon:
            theta_hat = np.concatenate(
                [
                    theta[:,i][:, np.newaxis]
                    for i in np.arange(n_actions)
                ],
                axis=1,
            )  # dim * n_actions
            predicted_rewards = (context @ theta_hat).flatten()
            return predicted_rewards.argsort()[::-1][: len_list]
        else:
            return self.random_.choice(
                n_actions, size=len_list, replace=False
            )


@dataclass
class LinUCB(BaseLinPolicy):
    """Linear Upper Confidence Bound.
    Parameters
    ----------
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
    epsilon: float, default=0.
        Exploration hyperparameter that must be greater than or equal to 0.0.
    References
    --------------
    L. Li, W. Chu, J. Langford, and E. Schapire.
    A contextual-bandit approach to personalized news article recommendation.
    In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010.
    """

    def __init__(self,epsilon=0.2) -> None:
        """Initialize class."""
        self.epsilon = epsilon
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0)
        self.policy_name = f"linear_ucb_{self.epsilon}"

        super().__init__()

    def get_action(self, context: np.ndarray,len_list,model_id=None) -> np.ndarray:
        """Select action for new data.
        Parameters
        ----------
        context: array
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
        model = self._model_storage.get_model(model_id)
        A_inv = model['A_inv']  # pylint: disable=invalid-name
        theta = model['theta_hat']

        check_array(array=context, name="context", expected_dim=2)
        if context.shape[0] != 1:
            raise ValueError("Expected `context.shape[0] == 1`, but found it False")

        context_dim,n_actions = theta.shape

        theta_hat = np.concatenate(
            [
                theta[:,i][:, np.newaxis]
                for i in np.arange(n_actions)
            ],
            axis=1,
        )  # dim * n_actions
        sigma_hat = np.concatenate(
            [
                np.sqrt(context @ A_inv[i] @ context.T)
                for i in np.arange(n_actions)
            ],
            axis=1,
        )  # 1 * n_actions
        ucb_scores = (context @ theta_hat + self.epsilon * sigma_hat).flatten()
        return ucb_scores.argsort()[::-1][: len_list]


@dataclass
class LinTS(BaseLinPolicy):
    """Linear Thompson Sampling.
    Parameters
    ----------
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
    """

    def __post_init__(self) -> None:
        """Initialize class."""
        self.policy_name = "linear_ts"
        
        super().__init__()

    def select_action(self, context: np.ndarray,len_list,model_id=None) -> np.ndarray:
        """Select action for new data.
        Parameters
        ----------
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
        model = self._model_storage.get_model(model_id)
        A_inv = model['A_inv']  # pylint: disable=invalid-name
        theta = model['theta_hat']
        context_dim,n_actions = theta.shape
        theta_hat = np.concatenate(
            [
                theta[:, i][:, np.newaxis]
                for i in np.arange(n_actions)
            ],
            axis=1,
        )
        theta_sampled = np.concatenate(
            [
                self.random_.multivariate_normal(theta_hat[:, i], A_inv[i])[
                    :, np.newaxis
                ]
                for i in np.arange(n_actions)
            ],
            axis=1,
        )

        predicted_rewards = (context @ theta_sampled).flatten()
        return predicted_rewards.argsort()[::-1][: len_list]