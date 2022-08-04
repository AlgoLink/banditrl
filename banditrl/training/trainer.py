from typing import Dict

import pandas as pd
import os

from ..utils import feature_importance, model_constructors, utils
from ..preprocessing import preprocessor
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


def train(
    training_df: pd.DataFrame,
    ml_config: Dict = None,
    feature_config: Dict = None,
    predictor_save_dir: str = None
):
    utils.validate_ml_config(ml_config)
    logger.info("检查训练数据的格式...")
    utils.validate_training_data_schema(training_df)
    if len(training_df) == 0:
        logger.error(f"没有发现训练数据的记录，训练中止。")
        return None

    logger.info(f"获取行数为 {len(training_df)} 的训练数据。")
    logger.info(training_df.head())
    utils.fancy_print("Kicking off data preprocessing")
    if ml_config["features"].get("context_free", False):
        data = training_df
    else:
        # always add decision as a feature to use if not using all features
        features_to_use = ml_config["features"].get("features_to_use", ["*"])
        if features_to_use != ["*"]:
            features_to_use.append(preprocessor.DECISION_FEATURE_NAME)
        features_to_use = list(set(features_to_use))
        dense_features_to_use = ml_config["features"].get("dense_features_to_use", ["*"])
    
        data = preprocessor.preprocess_data(
            training_df,
            feature_config,
            ml_config["reward_type"],
            features_to_use,
            dense_features_to_use,
        )
        X, y = preprocessor.data_to_pytorch(data)
    model_type = ml_config["model_type"]
    model_params = ml_config["model_params"][model_type]
    reward_type = ml_config["reward_type"]
    
    feature_importance_params = ml_config.get("feature_importance", {})
    if feature_importance_params.get("calc_feature_importance", False):
        # calculate feature importances - only works on non id list features at this time
        utils.fancy_print("Calculating feature importances")
        feature_scores = feature_importance.calculate_feature_importance(
            reward_type=reward_type,
            feature_names=data["final_float_feature_order"],
            X=X,
            y=y,
        )
        feature_importance.display_feature_importances(feature_scores)

        # TODO: Make keeping the top "n" features work in predictor. Right now
        # using this feature breaks predictor, so don't use it in a final model,
        # just use it to experiment in seeing how model performance is.
        if feature_importance_params.get("keep_only_top_n", False):
            utils.fancy_print("Keeping only top N features")
            X, final_float_feature_order = feature_importance.keep_top_n_features(
                n=feature_importance_params["n"],
                X=X,
                feature_order=data["final_float_feature_order"],
                feature_scores=feature_scores,
            )
            data["final_float_feature_order"] = final_float_feature_order
            logger.info(f"保留前 {feature_importance_params['n']} 的特征:")
            logger.info(final_float_feature_order)

    utils.fancy_print("Starting training")
    # build the model
    # model storage
    storage = ml_config["storage"]
    if storage["model"].get("type","rlite")=="rlite":
        dbpath = storage["model"].get("path",os.path.join(os.getcwd(),"model.db"))
        model_storage=RliteModelStorage(dbpath)
        if ml_config["features"].get("context_free", False):
            rlite_path = dbpath
    elif storage["model"].get("type","rlite")=="redis":
        host= storage["model"].get("host",'0.0.0.0')
        port= storage["model"].get("port",6379)
        db= storage["model"].get("db",1)
        model_storage=RedisModelStorage(host,port,db)
    else:
        model_storage = MemoryModelStorage()
    # his context storage
    if storage["his_context"].get("type","rlite")=="rlite":
        dbpath = storage["his_context"].get("path",os.path.join(os.getcwd(),"his_context.db"))
        his_context_storage=RliteHistoryStorage(dbpath)
    else:
        his_context_storage = MemoryHistoryStorage()
        
    # action storage
    if storage["action"].get("type","rlite")=="rlite":
        dbpath = storage["action"].get("path",os.path.join(os.getcwd(),"action.db"))
        action_storage=RliteActionStorage(dbpath)
    else:
        action_storage = MemoryActionStorage()
    
    return X,y,data
    