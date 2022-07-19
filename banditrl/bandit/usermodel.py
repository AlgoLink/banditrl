from typing import List

import hirlite
import pickle

import random
import numpy as np
import redis
import pickle

from dataclasses import dataclass
from sklearn.utils import check_scalar
from sklearn.utils import check_random_state

import logging

import six

from banditrl.bandit.base import BaseBandit
from .utils import check_array

# configurations to replicate the Bernoulli Thompson Sampling policy used in ZOZOTOWN production
#prior_bts_file = os.path.join(os.path.dirname(__file__), "conf", "prior_bts.yaml")
#with open(prior_bts_file, "rb") as f:
#    production_prior_for_bts = yaml.safe_load(f)

DEFAULT_PREFIX = "banditrl:ee"

def _key(k):
    return "{0}:{1}".format(DEFAULT_PREFIX, k)


class RliteEE:
    def __init__(self,
                 rlite_path=None,
                 model_id=None):
        if rlite_path is None:
            self.rlite_client= hirlite.Rlite("online_rlite.db",encoding='utf8')
        else:
            self.rlite_client= hirlite.Rlite(rlite_path,encoding='utf8')
        self._model_id = model_id


    def _increment_model_tries(self, 
                               model: str, 
                               uid: str, 
                               model_id: str) -> None:
        key_tries= _key("{0}:{1}:{2}:tries".format(uid,model_id,model))
        model_tries = self.rlite_client.command("incr",key_tries)
        return model_tries

    def get_mostpop(self,uid,model_id,topN,withscores=False):
        score_key = _key("{0}:{1}:score".format(uid,model_id))
        if withscores:
            score_list = self.rlite_client.command("zrange",
                                                   score_key,
                                                   "0",
                                                   str(topN-1),
                                                   "withscores")
        else:
            score_list = self.rlite_client.command("zrange",
                                                   score_key,
                                                   "0",
                                                   str(topN-1))
        return score_list

    def _epsilon_greedy_selection(self,uid, model_id,topN):
        models = self.get_models(model_id)
        epsilon = self.get_epsilon(model_id)
        random_recom = self.get_random_signal(model_id)

        if random_recom is None:
            random_recom = 1.0
        else:
            random_recom = float(random_recom)
        if epsilon is None:
            epsilon = 0.2
        else:
            epsilon = float(epsilon)
            
        if random_recom>0:
            if random.random() < epsilon:
                if topN > len(models):
                    res = random.sample(models, len(models))
                else:
                    res = random.sample(models, topN)
                recommendation_ids = res
            else:
                res = self.get_mostpop(uid,model_id,topN)
                recommendation_ids = res
        else:
            res = self.get_mostpop(uid,model_id,topN)
            recommendation_ids = res
            
        return recommendation_ids

    def expose_actions(self,uid,items_list,model_id):
        for item_id in items_list:
            #更新曝光ID的 score
            item_id = str(item_id)
            model_tries = self._increment_model_tries(item_id, uid, model_id)
            success_key = _key("{0}:{1}:{2}:reward_successes".format(uid,model_id,item_id))
            _reward = self.rlite_client.command("get",success_key)
            if _reward is None:
                _reward = 0.0
            else:
                _reward = float(_reward)
            _model_score = _reward/(model_tries + 0.00000001)
            score_key = _key("{0}:{1}:score".format(uid,model_id))
            model_score="-{}".format(_model_score)
            self.rlite_client.command("zadd",score_key, model_score,str(item_id))

        return True

    def select_model(self,uid,model_id,topN=3,ifexpose="yes") -> str:
        epsilon_greedy_selection = self._epsilon_greedy_selection(uid,model_id,topN=topN)
        if ifexpose=="yes":
            self.expose_actions(uid, epsilon_greedy_selection,model_id)

        return epsilon_greedy_selection

    def reward_model(self, 
                     model: str,
                     uid:str,
                     model_id:str,
                     reward:float=None,
                     init_model="no") -> None:

        success_key = _key("{0}:{1}:{2}:reward_successes".format(uid,model_id,model))
        score_key = _key("{0}:{1}:score".format(uid,model_id))
        key_tries=_key("{0}:{1}:{2}:tries".format(uid,model_id,model))

        if reward is None:
            reward = 1.0
        _reward = self.rlite_client.command("get",success_key)
        if _reward is None:
            _reward = 0.0
        else:
            _reward = float(_reward)
        _reward+= reward
        self.rlite_client.command("set",success_key,str(_reward))

        model_tries = self.rlite_client.command("get",key_tries)
        if model_tries is None:
            model_tries = 1 
        else:
            model_tries = int(model_tries)

        if init_model == 'yes':
            # 初始化模型时默认曝光
            self._increment_model_tries(model,uid,model_id)

        _model_score = _reward/float(model_tries)
        model_score="-{}".format(_model_score)
        self.rlite_client.command("zadd",score_key, model_score,str(model))

    def set_epsilon(self,
                    model_id:str=None,
                    epsilon:float=0.2):
        epsilon_key = _key("{0}:epsilon".format(model_id))
        self.rlite_client.command("set",epsilon_key,str(epsilon))
        return True
    def get_epsilon(self,model_id):
        epsilon_key = _key("{0}:epsilon".format(model_id))
        return self.rlite_client.command("get",epsilon_key)

    def set_random_signal(self,
                    model_id:str=None,
                    ifrandom:float=0.0):
        random_key = _key("{0}:randomrecom".format(model_id))
        self.rlite_client.command("set",random_key,str(ifrandom))
        return True
    def get_random_signal(self,model_id):
        random_key = _key("{0}:randomrecom".format(model_id))
        return self.rlite_client.command("get",random_key)
    def add_model(self,model_id:str=None,model:str=None):
        # model = item_id
        key_models = _key("{0}:models".format(model_id) )
        self.rlite_client.command("sadd",key_models,str(model))

        return True

    def get_models(self,model_id:str=None):
        key_models = _key("{0}:models".format(model_id))
        models = self.rlite_client.command("smembers",key_models)
        return models

class BTS(BaseBandit):
    """Bernoulli Thompson Sampling Policy
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
    alpha: array-like, shape (n_actions, ), default=None
        Prior parameter vector for Beta distributions.
    beta: array-like, shape (n_actions, ), default=None
        Prior parameter vector for Beta distributions.
    is_zozotown_prior: bool, default=False
        Whether to use hyperparameters for the beta distribution used
        at the start of the data collection period in ZOZOTOWN.
    campaign: str, default=None
        One of the three possible campaigns considered in ZOZOTOWN, "all", "men", and "women".
    policy_name: str, default='bts'
        Name of bandit policy.
    """
    is_prior: bool = False
    policy_name: str = "bts"
    def __init__(self,
                 history_storage,
                 model_storage,
                 action_storage,
                 recommendation_cls=None,
                 random_state=2022,
                 action_list=[],
                 alpha=1,
                 beta = 1,
                 model_id=None,
                 campaign: Optional[str] = None):
        super(BTS, self).__init__(history_storage, 
                                  model_storage,
                                  action_storage,
                                  recommendation_cls)
        self._model_id=model_id
        self.random_ = check_random_state(random_state)
        self.production_prior_for_bts = {}
        self.campaign = campaign
        self.action_list=action_list
        if self.is_prior:
            self.alpha = production_prior_for_bts[self.campaign]["alpha"]
            self.beta = production_prior_for_bts[self.campaign]["beta"]
        else:
            self.alpha = alpha
            self.beta = beta
        self.policy_name = f"bandit_{self.policy_name}"
        
        model = self._model_storage.get_model(self._model_id)
        if model is None:
            self._init_model()

    def _init_model(self,model_id=None):
        if model_id is None:
            model_id = self._model_id
        # model init
        
        for action in self.action_list:
            model = {}
            model_key="action:{0}:model:{1}".format(action,model_id)
            alpha_key="action:{0}:alpha".format(action)
            beta_key="action:{0}:beta".format(action)
            tries_key="action:{0}:tries".format(action)
            reward_key = "action:{0}:rewards".format(action)
            model[tries_key]=0
            model[alpha_key] =1
            model[beta_key] =1
            model[reward_key] =0
            self._model_storage.save_model(model,model_key)

    def get_action(self,topN=1,model_id=None):
        """Select a list of actions.
        Returns
        ----------
        selected_actions: array-like, shape (len_list, )
            List of selected actions.
        """
        if model_id is None:
            model_id = self._model_id

        score_key = "modelscore:bts:{0}".format(model_id)
        predicted_rewards = self._model_storage.rlite_client.command("zrange",
                                                                     score_key,
                                                                     "0",
                                                                     str(topN-1))
        self.expose_actions(predicted_rewards,model_id=model_id)
        return predicted_rewards

    def reward(self, 
               action: int, 
               reward: float,
               model_id:str =None,
               offline= False) -> None:
        """Update policy parameters.
        Parameters
        ----------
        action: int
            Selected action by the policy.
        reward: float
            Observed reward for the chosen action and position.
        """
        if model_id is None:
            model_id = self._model_id
        model_key="action:{0}:model:{1}".format(action,model_id)
        model = self._model_storage.get_model(model_key)
        tries_key="action:{0}:tries".format(action)
        alpha_key="action:{0}:alpha".format(action)
        beta_key="action:{0}:beta".format(action)
        reward_key = "action:{0}:rewards".format(action)
        model_tries = model[tries_key]
        rewards = model[reward_key]
        beta = model[beta_key]
        alpha = model[alpha_key]
        if offline:
            model_tries+=1
            
        rewards+=reward
        model[reward_key] = rewards
        model[tries_key] = model_tries
        
        _b = (model_tries - rewards) + beta
        if _b<=0:
            _b = 1
        _model_score = self.random_.beta(
                a=rewards + alpha,
                b= _b,
            )
        model_score="-{}".format(_model_score)
        self._model_storage.rlite_client.command("zadd",score_key, model_score,str(action))
        self._model_storage.save_model(model,model_key)


    def expose_actions(self,actions_list,model_id=None):
        if model_id is None:
            model_id = self._model_id

        score_key = "modelscore:bts:{0}".format(model_id)
        for action in actions_list:
            model_key="action:{0}:model:{1}".format(action,model_id)
            tries_key="action:{0}:tries".format(action)
            alpha_key="action:{0}:alpha".format(action)
            beta_key="action:{0}:beta".format(action)
            reward_key = "action:{0}:rewards".format(action)
            
            model = self._model_storage.get_model(model_key)
            #更新曝光ID的 score
            model_tries = model.get(tries_key)
            alpha = model[alpha_key]
            beta = model[beta_key]
            rewards = model[reward_key]
            if model_tries is None:
                model_tries = 1
            else:
                model_tries = int(model_tries)+1
            model[tries_key] = model_tries
            _b = (model_tries - rewards) + beta
            if _b<=0:
                _b = 1
            _model_score = self.random_.beta(
                a=rewards + alpha,
                b= _b,
            )
            model_score="-{}".format(_model_score)
            self._model_storage.rlite_client.command("zadd",score_key, model_score,str(action))
            
            self._model_storage.save_model(model,model_key)