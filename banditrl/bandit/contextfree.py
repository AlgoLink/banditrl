"""Context-Free Bandit Algorithms."""
from dataclasses import dataclass
import os
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar
from sklearn.utils import check_random_state

from banditrl.bandit.base import BaseBandit
from .utils import check_array

@dataclass
class EpsilonGreedy(BaseBandit):
    """Epsilon Greedy policy.
    Parameters
    ----------
    n_actions: int
        Number of actions.
    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.
    batch_size: int, default=1
        Number of samples used in a batch parameter update.
    random_state: int, default=None
        Controls the random seed in sampling actions.
    epsilon: float, default=1.
        Exploration hyperparameter that must take value in the range of [0., 1.].
    policy_name: str, default=f'egreedy_{epsilon}'.
        Name of bandit policy.
    """

    def __init__(self,
                 history_storage,
                 model_storage,
                 action_storage,
                 recommendation_cls=None,
                 n_actions=None,
                 random_state=2022,
                 epsilon = 1.0,
                 model_id=None):
        super(LinEpsilonGreedy, self).__init__(history_storage, 
                                            model_storage,
                                            action_storage,
                                            recommendation_cls)
        """Initialize Class."""
        self.random_state = random_state
        self.random_ = check_random_state(self.random_state)
        self.epsilon = epsilon
        self._model_id = model_id
        check_scalar(self.epsilon, "epsilon", float, min_val=0.0, max_val=1.0)
        self.policy_name = f"egreedy_{self.epsilon}"

    def get_action(self,model_id=None) -> np.ndarray:
        """Select a list of actions.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        if model_id is None:
            model_id = self._model_id
        if (self.random_.rand() > self.epsilon) and (self.action_counts.min() > 0):
            predicted_rewards = self.reward_counts / self.action_counts
            return predicted_rewards.argsort()[::-1][: self.len_list]
        else:
            return self.random_.choice(
                self.n_actions, size=self.len_list, replace=False
            )

    def reward(self, action: int, reward: float) -> None:
        """Update policy parameters.
        Parameters
        ----------
        action: int
            Selected action by the policy.
        reward: float
            Observed reward for the chosen action and position.
        """
        self.n_trial += 1
        self.action_counts_temp[action] += 1
        self.reward_counts_temp[action] += reward
        if self.n_trial % self.batch_size == 0:
            self.action_counts = np.copy(self.action_counts_temp)
            self.reward_counts = np.copy(self.reward_counts_temp)
