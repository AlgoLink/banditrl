from ..preprocessing import preprocessor
from ..preprocessing.feature_encodings import BanditContextEncoder
from ..utils import model_constructors,utils
import os
import json
import hirlite
import pickle

from ..storage import (
    MemoryHistoryStorage,
    RliteHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    RliteActionStorage,
    RliteModelStorage,
    Action
    
)

logger = utils.get_logger(__name__)

class BanditPredictor:
    """Class used to make predictions given a trained bandit model including reward."""
    def __init__(self,ml_config={}):
        self.ml_config = ml_config
        self.model=self.build_model
        self.model_type=ml_config.get("model_type")
        self.model_id = ml_config.get("model_id")
        self.reward_type = ml_config.get("reward_type","regression")
        log_path = ml_config["storage"].get("log_path")
        model_acc_path = ml_config["storage"].get("model_acc_path")
        if log_path:
            self.log_db = utils.log_data(log_path)
        else:
            self.log_db = None
        if model_acc_path:
            self.model_acc = hirlite.Rlite(os.path.join(model_acc_path,"model_acc.db"),encoding='utf8')
        else:
            self.model_acc = None
        predictor_save_dir = self.ml_config["storage"].get("predictor_save_dir")
        if predictor_save_dir is not None:
            self.feature_transformer = self.build_feature_transformer

    @property
    def build_model(self):
        model_type = self.ml_config["model_type"]
        model_id = self.ml_config.get("model_id")
        model_params = self.ml_config["model_params"][model_type]
        reward_type = self.ml_config["reward_type"]
        predictor_save_dir = self.ml_config["storage"].get("predictor_save_dir")
        if predictor_save_dir is not None:
            logger.info("loading model saved meta from disk...")
            model_id = self.ml_config.get("model_id", "model")
            save_dir = f"{predictor_save_dir}/{model_id}/"
            model_meta_path = f"{save_dir}/{model_id}_meta.json"
            with open(model_meta_path, "rb") as f:
                self.model_meta = json.load(f)
            self.action_to_itemid=self.model_meta.get("action_to_itemid")
            self.itemid_to_action=self.model_meta.get("itemid_to_action")
            self.context_dim = self.model_meta.get("context_dim")
            self.n_actions = self.model_meta.get("n_actions")
        # model storage
        storage = self.ml_config["storage"]
        if storage["model"].get("type","rlite")=="rlite":
            modelpath = storage["model"].get("path",os.getcwd())
            model_db = os.path.join(modelpath,f"{model_id}_model.db")
            model_storage= RliteModelStorage(model_db)
            if self.ml_config["features"].get("context_free", False):
                rlite_path = model_db
        elif storage["model"].get("type","rlite")=="redis":
            host= storage["model"].get("host",'0.0.0.0')
            port= storage["model"].get("port",6379)
            db= storage["model"].get("db",1)
            model_storage= RedisModelStorage(host,port,db)
        else:
            model_storage = MemoryModelStorage()
        # his context storage
        if storage["his_context"].get("type","rlite")=="rlite":
            dbpath = storage["his_context"].get("path",os.getcwd())
            his_db = os.path.join(dbpath,f"{model_id}_his_context.db")
            his_context_storage= RliteHistoryStorage(his_db)
        else:
            his_context_storage = MemoryHistoryStorage()
        
        # action storage
        if storage["action"].get("type","mem")=="rlite":
            dbpath = storage["action"].get("path",os.path.join(os.getcwd(),"action.db"))
            action_storage=RliteActionStorage(dbpath)
        else:
            action_storage = MemoryActionStorage()
            if not self.ml_config["features"].get("context_free", False):
                n_actions = self.model_meta.get("n_actions")
                action_storage.add([Action(i) for i in range(n_actions)])
        self.action_storage = action_storage
        
        if model_type == "rliteee":
            model = model_constructors.build_rliteee_model(rlite_path,model_id=model_id)
        
        elif model_type=="bts":
            model = model_constructors.build_bts_model(his_context_storage,
                                                       model_storage,
                                                       alpha= model_params.get("alpha",1), 
                                                       beta = model_params.get("beta",1),
                                                       model_id= model_id)

        elif model_type == "linucb_array":
            _n_actions = model_params.get("n_actions") or n_actions
            _dim = model_params.get("context_dim") or self.context_dim
            model = model_constructors.build_linucb_array_model(his_context_storage,
                                                                model_storage,
                                                                action_storage,
                                                                n_actions = _n_actions,
                                                                context_dim = _dim,
                                                                alpha=model_params.get("alpha",0.2),
                                                                model_id=model_id)
        elif model_type == "linucb_dict":
            _dim = model_params.get("context_dim") or self.context_dim
            model = model_constructors.build_linucb_dict_model(his_context_storage,
                                                               model_storage,
                                                               action_storage,
                                                               context_dim = _dim,
                                                               alpha= model_params.get("alpha",0.2),
                                                               model_id= model_id)
            
        elif model_type == "logisticucb":
            _n_actions = model_params.get("n_actions") or n_actions
            _dim = model_params.get("context_dim") or self.context_dim
            model = model_constructors.build_logisticucb_model(his_context_storage,
                                                               model_storage,
                                                               action_storage,
                                                               n_actions = _n_actions,
                                                               context_dim=_dim,
                                                               epsilon =model_params.get("epsilon",0.2),
                                                               alpha_ = model_params.get("alpha",1.0),
                                                               lambda_ = model_params.get("lambda",1.0),
                                                               model_id= model_id)
        return model

    @property
    def build_feature_transformer(self):
        predictor_save_dir = self.ml_config["storage"].get("predictor_save_dir")
        if predictor_save_dir is not None:
            logger.info("loading predictor artifacts from disk...")
            model_id = self.ml_config.get("model_id", "model")
            save_dir = f"{predictor_save_dir}/{model_id}/"
            predictor_config_path = f"{save_dir}/{model_id}_features.pkl"
            feature_transformer = BanditContextEncoder.encoder_from_file(predictor_config_path)
            return feature_transformer.predict
        else:
            return None
        
    def get_action(self,
                   feature={}, 
                   request_id=None,
                   model_id=None,
                   topN=1,
                   uid=None,
                   auto_feature=True,
                   user_model=True):
        if self.model_type in ("linucb_array","logisticucb"):
            if auto_feature:
                features = self.feature_transformer(feature)["pytorch_input"]["X_float"].numpy()
            else:
                features = feature
            recoms = self.model.get_action(features,topN,request_id,model_id)
            try:
                recom_list=[self.action_to_itemid.get(str(i.action.id)) for i in recoms]
            except:
                recom_list=[self.action_to_itemid.get(i.action.id) for i in recoms]
        elif self.model_type=="linucb_dict":
            if auto_feature:
                features = self.feature_transformer(feature)["pytorch_input"]["X_float"].numpy()
            else:
                features = feature
            _context = {action_id: features for action_id in self.action_storage.iterids()}
            _,recoms = self.model.get_action(_context,topN,request_id,model_id)

            try:
                recom_list=[self.action_to_itemid.get(str(i.action.id)) for i in recoms]
            except:
                recom_list=[self.action_to_itemid.get(i.action.id) for i in recoms]
                
        elif self.model_type == "rliteee":
            recom_list = self.model.select_model(uid=uid,model_id=model_id,topN=topN)
        elif self.model_type == "bts":
            if user_model:
                uid_model=f"{uid}_{model_id}"
                recom_list = self.model.get_action(topN= topN,model_id= uid_model)
            else:
                recom_list = self.model.get_action(topN= topN,model_id= model_id)

        if self.model_acc:
            request_key = f"{model_id}_{request_id}_{self.model_type}"
            model_acc_expose_key=f"{model_id}_{self.model_type}_expose"
            self.model_acc.incr(model_acc_expose_key)
            self.model_acc.incr(request_key)
            
            
        if self.log_db is not None:
            if len(recom_list)>0:
                recom = str(recom_list[0])
            else:
                recom = ""
            measurement = "predict"
            tags = {"model_id":str(model_id),
                    "request_id":str(request_id),
                    "model_type":self.model_type,
                    "recom":recom}
            fields = {"topN":float(topN),"log_type": 4}
            self.log_db.log_model_details(measurement,tags,fields)
        return recom_list
    
    def reward(self,request_id,action,reward, model_id,uid=None,user_model=True):
        if self.model_type in ("linucb_array","logisticucb"):
            actionid=self.itemid_to_action.get(action)
            if self.reward_type=='binary':
                reward = 1.0
            self.model.reward(request_id, 
                              int(actionid), 
                              float(reward),
                              model_id=model_id)
        elif self.model_type=="linucb_dict":
            actionid=self.itemid_to_action.get(action)
            rewards = {int(actionid):float(reward)}
            self.model.reward(history_id=request_id,
                              rewards=rewards,
                              model_id=model_id)
        elif self.model_type=="rliteee":
            self.model.reward_model(model=action,
                                    uid=uid,
                                    model_id=model_id,
                                    reward=reward)
        elif self.model_type=="bts":
            if user_model:
                uid_model=f"{uid}_{model_id}"
            else:
                uid_model = model_id
            self.model.reward(action = action,
                              reward=reward,
                              model_id=uid_model)
            
        if self.model_acc:
            model_types_key = f"{model_id}_model_types"
            self.model_acc.sadd(model_types_key,self.model_type)
            request_key = f"{model_id}_{request_id}_{self.model_type}"
            model_acc_expose_key=f"{model_id}_{self.model_type}_expose"
            model_acc_reward_key=f"{model_id}_{self.model_type}_reward"
            model_acc_key=f"{model_id}_{self.model_type}"
            reward_key = f"{model_id}_{self.model_type}_rewardscum"

            if self.model_acc.get(request_key):
                #model_acc_expose = self.model_acc.get(model_acc_expose_key)
                #if model_acc_expose:
                #    model_acc_reward = self.model_acc.incr(model_acc_reward_key)
                #    model_acc = float(model_acc_reward)/float(model_acc_expose)
                #    self.model_acc.set(model_acc_key,str(model_acc))
                rewardscum = self.model_acc.get(reward_key)
                if rewardscum:
                    _rewardscum = float(rewardscum)
                    _rewardscum+= float(reward)
                    self.model_acc.set(reward_key,str(_rewardscum))
                else:
                    self.model_acc.set(reward_key,str(reward))

        if self.log_db is not None:
            measurement = "reward"
            tags = {"model_id":str(model_id),
                    "request_id":str(request_id),
                    "model_type":self.model_type,
                    "recom":str(action)}
            fields = {"reward":float(reward),"log_type":2}
            self.log_db.log_model_details(measurement,tags,fields)
        return True
    
    def get_model_acc(self,model_id):
        model_acc_expose_key=f"{model_id}_{self.model_type}_expose"
        model_acc_reward_key=f"{model_id}_{self.model_type}_reward"

        reward_key = f"{model_id}_{self.model_type}_rewardscum"
        model_acc_reward=self.model_acc.get(model_acc_reward_key)
        model_acc_expose = self.model_acc.get(model_acc_expose_key)

        if model_acc_reward and model_acc_expose:
            model_acc = float(model_acc_reward)/float(model_acc_expose)
        else:
            model_acc = 0.0
        model_reward_cum = self.model_acc.get(reward_key)
        model_types_key = f"{model_id}_model_types"
        model_types = self.model_acc.smembers(model_types_key)
        result = {"model_acc":model_acc,
                  "model_reward_cum":model_reward_cum,
                  "model_type":self.model_type,
                  "model_types":model_types}
        
        return result
        