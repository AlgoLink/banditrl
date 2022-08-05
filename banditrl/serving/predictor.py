from ..preprocessing import preprocessor
from ..preprocessing.feature_encodings import BanditContextEncoder
from ..utils import model_constructors,utils
import os

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
    def __init__(self,ml_config,predictor_save_dir: str = None):
        self.ml_config = ml_config
        if not self.ml_config["features"].get("context_free", False):
            if predictor_save_dir is not None:
                logger.info("loading predictor artifacts from disk...")
                model_id = self.ml_config.get("model_id", "model")
                
                save_dir = f"{predictor_save_dir}/{model_id}/"
                predictor_config_path = f"{save_dir}/{model_id}_features.pkl"
                self.feature_transformer = BanditContextEncoder.encoder_from_file(predictor_config_path)
    @property
    def build_model(self):
        model_type = self.ml_config["model_type"]
        model_id = self.ml_config.get("model_id", None)
        model_params = self.ml_config["model_params"][model_type]
        reward_type = self.ml_config["reward_type"]
        # model storage
        storage = self.ml_config["storage"]
        if storage["model"].get("type","rlite")=="rlite":
            dbpath = storage["model"].get("path",os.path.join(os.getcwd(),"model.db"))
            model_storage= RliteModelStorage(dbpath)
            if self.ml_config["features"].get("context_free", False):
                rlite_path = dbpath
        elif storage["model"].get("type","rlite")=="redis":
            host= storage["model"].get("host",'0.0.0.0')
            port= storage["model"].get("port",6379)
            db= storage["model"].get("db",1)
            model_storage= RedisModelStorage(host,port,db)
        else:
            model_storage = MemoryModelStorage()
        # his context storage
        if storage["his_context"].get("type","rlite")=="rlite":
            dbpath = storage["his_context"].get("path",os.path.join(os.getcwd(),"his_context.db"))
            his_context_storage= RliteHistoryStorage(dbpath)
        else:
            his_context_storage = MemoryHistoryStorage()
        
        # action storage
        if storage["action"].get("type","rlite")=="rlite":
            dbpath = storage["action"].get("path",os.path.join(os.getcwd(),"action.db"))
            action_storage=RliteActionStorage(dbpath)
        else:
            action_storage = MemoryActionStorage()
        
        if model_type == "rliteee":
            model = model_constructors.build_rliteee_model(rlite_path,model_id=model_id)
        
        elif model_type=="bts":
            model = model_constructors.build_bts_model(his_context_storage,
                                                       model_storage,
                                                       alpha= model_params.get("alpha",1), 
                                                       beta = model_params.get("beta",1),
                                                       model_id= model_id)

        elif model_type == "linucb_array":
            model = model_constructors.build_linucb_array_model(his_context_storage,
                                                                model_storage,
                                                                action_storage,
                                                                n_actions = model_params.get("n_actions"),
                                                                context_dim = model_params.get("context_dim"),
                                                                alpha=model_params.get("alpha",0.2),
                                                                model_id=model_id)
        elif model_type == "linucb_dict":
            model = model_constructors.build_linucb_dict_model(his_context_storage,
                                                               model_storage,
                                                               action_storage,
                                                               context_dim = model_params.get("context_dim"),
                                                               alpha= model_params.get("alpha",0.2),
                                                               model_id= model_id)
            
        return model