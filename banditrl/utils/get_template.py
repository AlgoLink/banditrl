from pathlib import Path
from ..ext.tinydb import TinyDB
import os

TEMPLATE_FOLDER_PATH = Path(__file__).parent.parent / "templates"

def config_template_gen(config_type="ml"):
    assert config_type in ("ml","feature")
    config_contents={}
    if config_type=="ml":
        config_contents["model_id"]="model_v1"
        config_contents["data_reader"] = {
            "features_to_use": ["*"],
            "dense_features_to_use": ["*"]}
        config_contents["feature_importance"]={
            "calc_feature_importance": False,
            "keep_only_top_n": True,
            "n": 10
        }
        config_contents["model_type"]="linear_bandit"
        config_contents["reward_type"] = "binary"
        config_contents["reward_types"] = "binary,regression"
        config_contents["model_params"]  ={ "linear_bandit":{"context_dim":None, "n_actions":2}}
        config_contents["train_percent"] = 0.8
        config_contents["version"] = 1.0
        config_contents["config_type"]="ml"
    elif config_type=="feature":
        config_contents["config_type"]="feature"
        config_contents["version"] = 1.0
        config_contents["model_id"] = "model_v1"
        config_contents["choices"] = ["male", "female"]
        config_contents["features"] = {
            "country": {"type": "P", "product_set_id": "1", "use_dense": False},
            "year": {"type": "N"},
            "decision": {"type": "C"},
        }
        config_contents["product_sets"]= {
            "1": {
                "ids": ["usa", "china", "india", "serbia", "norway"],
                "dense": {
                    "usa": ["north-america", 10.0],
                    "china": ["asia", 8.5],
                    "india": ["asia", 7.5],
                    "serbia": ["europe", 11.5],
                    "norway": ["europe", 10.5],
                },
                "features": [
                    {"name": "region", "type": "C"},
                    {"name": "avg_shoe_size_m", "type": "N"},
                ],
            }
        }
        
        
    return config_contents
        
def save_config(configs,db_path=None):
    if db_path is None:
        db_path = os.path.join(os.getcwd(),"model_configs.json")
    db=TinyDB(db_path)
    db.insert(configs)
    return True