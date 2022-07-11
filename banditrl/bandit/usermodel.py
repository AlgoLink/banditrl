from typing import List

import hirlite
import pickle

import random
import numpy as np
import redis
import pickle


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
        epsilon_key = _key("{0}:epsilon".format(model_id))
        random_key = _key("{0}:randomrecom".format(model_id))
        
        epsilon = self.rlite_client.command("get",epsilon_key)
        random_recom = self.rlite_client.command("get",random_key)
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
            self.rlite_client.command("zadd",model_score,str(item_id))

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
        self.rlite_client.command("zadd",model_score,str(model))

    def set_epsilon(self,
                    model_id:str=None,
                    epsilon:float=0.2):
        epsilon_key = _key("{0}:epsilon".format(model_id))
        self.rlite_client.command("set",epsilon_key,str(epsilon))
        return True

    def add_model(self,model_id:str=None,model:str=None):
        # model = item_id
        key_models = "{0}:models".format(model_id) 
        self.rlite_client.command("sadd",key_models,str(model))

        return True

    def get_models(self,model_id:str=None):
        key_models = _key("{0}:models".format(model_id))
        models = self.rlite_client.command("smembers",key_models)
        return models
