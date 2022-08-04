from typing import Dict

import pandas as pd

from ..utils import feature_importance, model_constructors, utils
from ..preprocessing import preprocessor
from ..storage import (
    MemoryHistoryStorage,
    RliteHistoryStorage,
    MemoryModelStorage,
    MemoryActionStorage,
    RliteModelStorage,
    Action
)

def train(
    training_df: pd.DataFrame,
    ml_config: Dict,
    feature_config: Dict,
    predictor_save_dir: str = None
):
    utils.validate_ml_config(ml_config)
    logger.info("检查训练数据的格式...")
    utils.validate_training_data_schema(training_df)
    if len(training_df) == 0:
        logger.error(f"没有发现训练数据的记录，训练中止。")
        return None

    logger.info(f"获取 {len(training_df)} 训练数据的行数。")
    logger.info(training_df.head())
    utils.fancy_print("开启数据预处理")
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
    