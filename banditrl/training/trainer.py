from typing import Dict

import pandas as pd
import pickle
import os
import shutil
import time
import json

from ..utils import feature_importance, model_constructors, utils,offline_model_trainers
from ..preprocessing import preprocessor
from ..preprocessing import feature_encodings

from sklearn.model_selection import train_test_split

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
    itemid_to_action:Dict= {},
    offline_train = True
):
    start = time.time()
    if len(itemid_to_action)>0:
        action_to_itemid = dict(zip(itemid_to_action.values(), itemid_to_action.keys()))

    utils.validate_ml_config(ml_config)
    _dbpath = ml_config["storage"]["model"].get("path",os.path.join(os.getcwd(),"model.db"))
    _model_id = ml_config.get("model_id", None)
    logger.info(f"接下来训练的模型为: {_model_id}")
    logger.info(f"模型存储的路径: {_dbpath}")
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
        _,context_dim = X["X_float"].shape
    model_type = ml_config["model_type"]
    model_id = ml_config.get("model_id", None)
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
        modelpath = storage["model"].get("path",os.getcwd())
        model_db = os.path.join(modelpath,f"{model_id}_model.db")
        model_storage= RliteModelStorage(model_db)
        if ml_config["features"].get("context_free", False):
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
        if not ml_config["features"].get("context_free", False):
            _n_actions = model_params.get("n_actions")
            action_storage.add([Action(i) for i in range(_n_actions)])
        
    if model_type == "rliteee":
        model = model_constructors.build_rliteee_model(rlite_path,model_id=model_id)
        
    elif model_type=="bts":
        model = model_constructors.build_bts_model(his_context_storage,
                                                   model_storage,
                                                   alpha= model_params.get("alpha",1), 
                                                   beta = model_params.get("beta",1),
                                                   model_id= model_id)

    elif model_type == "linucb_array":
        _dim = model_params.get("context_dim") or context_dim
        model = model_constructors.build_linucb_array_model(his_context_storage,
                                                            model_storage,
                                                            action_storage,
                                                            n_actions = model_params.get("n_actions"),
                                                            context_dim = _dim,
                                                            alpha=model_params.get("alpha",0.2),
                                                            model_id=model_id)
    elif model_type == "linucb_dict":
        _dim = model_params.get("context_dim") or context_dim
        model = model_constructors.build_linucb_dict_model(his_context_storage,
                                                           model_storage,
                                                           action_storage,
                                                           context_dim = _dim,
                                                           alpha= model_params.get("alpha",0.2),
                                                           model_id= model_id)
        
        
    elif model_type == "logisticucb":
        _dim = model_params.get("context_dim") or context_dim
        model = model_constructors.build_logisticucb_model(his_context_storage,
                                                           model_storage,
                                                           action_storage,
                                                           n_actions = model_params.get("n_actions"),
                                                           context_dim=_dim,
                                                           epsilon =model_params.get("epsilon",0.2),
                                                           alpha_ = model_params.get("alpha",1.0),
                                                           lambda_ = model_params.get("lambda",1.0),
                                                           model_id= model_id)
    elif model_type == "linear_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "Linear models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_linear_model(
            reward_type=reward_type,
            penalty=model_params.get("penalty",0.8),
            alpha=model_params.get("alpha",0.2),
        )
        model_spec = None
    elif model_type == "gbdt_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "GBDT models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_gbdt(
            reward_type=reward_type,
            learning_rate=model_params.get("learning_rate",0.1),
            n_estimators=model_params.get("n_estimators",5),
            max_depth=model_params.get("max_depth",4),
        )
        model_spec = None
    elif model_type == "random_forest_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "Random forest models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_random_forest(
            reward_type=reward_type,
            n_estimators=model_params.get("n_estimators",5),
            max_depth=model_params.get("max_depth",4),
        )
        model_spec = None
    # build the predictor
    if ml_config["features"].get("context_free", False):
        predictor = model
    else:
        predictor = feature_encodings.BanditContextEncoder(
            feature_config=feature_config,
            float_feature_order=data["float_feature_order"],
            id_feature_order=data["id_feature_order"],
            id_feature_str_to_int_map=data["id_feature_str_to_int_map"],
            transforms=data["transforms"],
            imputers=data["imputers"],
            model= model,
            model_type= model_type,
            reward_type= reward_type,
            model_spec= None,
            dense_features_to_use=["*"],
        )
    # train the model
    if model_type == "rliteee":
        for index,rows in training_df.iterrows():
            #if ml_config.get("if_usermodel",True)
            uid = rows.get("uid",index)
            reward = rows.get("reward",1.0)
            decision = rows['decision']
            model.reward_model(model=decision,
                               uid=uid,
                               model_id = model_id,
                               reward = float(reward),
                               init_model="yes")
    elif model_type == "bts":
        for index,rows in training_df.iterrows():
            uid = rows.get("uid",index)
            reward = rows.get("reward",1.0)
            decision = rows['decision']
            if ml_config.get("if_usermodel",True):
                uid_model = "{0}_{1}".format(uid,model_id)
            else:
                uid_model = model_id
            model.reward(decision, reward=float(reward),model_id =uid_model,offline= offline_train)
    elif model_type == "linucb_array":
        train_percent = ml_config.get("train_percent",0.8)
        test_percent = 1- train_percent
        train, test = train_test_split(training_df, test_size= test_percent)
        logger.info(f"Training begin")
        for index,rows in training_df.iterrows():
            decision = rows['decision']
            context = X['X_float'][index].numpy().reshape(1,-1)
            
            reward = float(y[index])
            request_id = "{}_{}".format(index,model_id)
            model.get_action(context,len_list=1,request_id=request_id,model_id = model_id)
            action = itemid_to_action[decision]
            model.reward(request_id, int(action), float(reward),model_id=model_id)

        logger.info(f"Training end")
        logger.info(f"Testing begin")
        cum_reward = 0
        cum_correct=0
        cum_n_actions = 0
        for index,rows in test.iterrows():
            df_context = rows["context"]
            try:
                feats = json.loads(df_context)
            except:
                feats = df_context
            context = predictor.predict(feats, choices=None, get_ucb_scores=False, ext=False)
            context = context['pytorch_input']['X_float'].numpy()
            reward = float(rows["reward"])
            request_id = "{}_{}".format(index,model_id)            
            decision = rows['decision']
            request_id = "test_{}_{}".format(index,model_id)
            _context = {action_id: context for action_id in action_storage.iterids()}
            recom = model.get_action(context,len_list=1,request_id=request_id,model_id = model_id)
            predict_action = [action_to_itemid.get(int(i.action.id)) for i in recom][0]
            cum_n_actions +=1
            if str(predict_action)==str(decision):
                cum_reward+= reward
                cum_correct+=1
        avg_correct = cum_correct/cum_n_actions
        logger.info(utils.color_text(f"predictor's accuracy:{avg_correct}", color="blue"))
        logger.info(utils.color_text(f"all actions:{cum_n_actions}", color="blue"))
        logger.info(utils.color_text(f"cumulative rewards:{cum_reward}", color="blue"))
        
    elif model_type == "linucb_dict":
        train_percent = ml_config.get("train_percent",0.8)
        test_percent = 1- train_percent
        train, test = train_test_split(training_df, test_size= test_percent)
        logger.info(f"Training begin")
        for index,rows in train.iterrows():
            context = X['X_float'][index].numpy().tolist()
            reward = float(y[index])
            request_id = "{}_{}".format(index,model_id)            
            decision = rows['decision']
            request_id = "{}_{}".format(index,model_id)
            _context = {action_id: context for action_id in action_storage.iterids()}
            action = itemid_to_action[decision]
            model.get_action(_context, n_actions=1,request_id=request_id,model_id=model_id)
            model.reward(history_id=request_id, rewards={int(action):reward},model_id=model_id)
        logger.info(f"Training end")
        logger.info(f"Testing begin")
        cum_reward = 0
        cum_correct=0
        cum_n_actions = 0
        for index,rows in test.iterrows():
            df_context = rows["context"]
            try:
                feats = json.loads(df_context)
            except:
                feats = df_context
            context = predictor.predict(feats, choices=None, get_ucb_scores=False, ext=False)
            context = context['pytorch_input']['X_float'].numpy()
            reward = float(rows["reward"])
            request_id = "{}_{}".format(index,model_id)            
            decision = rows['decision']
            request_id = "test_{}_{}".format(index,model_id)
            _context = {action_id: context for action_id in action_storage.iterids()}
            _,recom = model.get_action(_context, n_actions=1,request_id=request_id,model_id=model_id)
            predict_action = [action_to_itemid.get(int(i.action.id)) for i in recom][0]
            cum_n_actions +=1
            if str(predict_action)==str(decision):
                cum_reward+= reward
                cum_correct+=1
        avg_correct = cum_correct/cum_n_actions
        logger.info(utils.color_text(f"predictor's accuracy:{avg_correct}", color="blue"))
        logger.info(utils.color_text(f"all actions:{cum_n_actions}", color="blue"))
        logger.info(utils.color_text(f"cumulative rewards:{cum_reward}", color="blue"))

    elif model_type == "logisticucb":
        train_percent = ml_config.get("train_percent",0.8)
        test_percent = 1- train_percent
        train, test = train_test_split(training_df, test_size= test_percent)
        logger.info(f"Training begin")
        for index,rows in training_df.iterrows():
            decision = rows['decision']
            context = X['X_float'][index].numpy().reshape(1,-1)
            
            reward = float(y[index])
            request_id = "{}_{}".format(index,model_id)

            model.get_action(context,len_list=1,request_id=request_id,model_id = model_id)
            action = itemid_to_action[decision]
            model.reward(request_id, int(action), float(reward),model_id=model_id)
            
        logger.info(f"Training end")
        logger.info(f"Testing begin")
        cum_reward = 0
        cum_correct=0
        cum_n_actions = 0
        for index,rows in test.iterrows():
            df_context = rows["context"]
            try:
                feats = json.loads(df_context)
            except:
                feats = df_context
            context = predictor.predict(feats, choices=None, get_ucb_scores=False, ext=False)
            context = context['pytorch_input']['X_float'].numpy()
            reward = float(rows["reward"])
            request_id = "{}_{}".format(index,model_id)            
            decision = rows['decision']
            request_id = "test_{}_{}".format(index,model_id)
            _context = {action_id: context for action_id in action_storage.iterids()}
            recom = model.get_action(context,len_list=1,request_id=request_id,model_id = model_id)
            predict_action = [action_to_itemid.get(int(i.action.id)) for i in recom][0]
            cum_n_actions +=1
            if str(predict_action)==str(decision):
                cum_reward+= reward
                cum_correct+=1
        avg_correct = cum_correct/cum_n_actions
        logger.info(utils.color_text(f"predictor's accuracy:{avg_correct}", color="blue"))
        logger.info(utils.color_text(f"all actions:{cum_n_actions}", color="blue"))
        logger.info(utils.color_text(f"cumulative rewards:{cum_reward}", color="blue"))

    elif model_type in ("gbdt_bandit", "random_forest_bandit", "linear_bandit"):
        logger.info(f"Training {model_type}")
        sklearn_model, _ = offline_model_trainers.fit_sklearn_model(
            reward_type=reward_type,
            model=model,
            X=X,
            y=y,
            train_percent=ml_config["train_percent"],
        )

    predictor_save_dir = ml_config["storage"].get("predictor_save_dir")
    if predictor_save_dir is not None:
        logger.info("Saving predictor artifacts to disk...")
        model_id = ml_config.get("model_id", "model")

        save_dir = f"{predictor_save_dir}/{model_id}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        predictor_net_path = f"{save_dir}/{model_id}_model.pt"
        predictor_config_path = f"{save_dir}/{model_id}_features.pkl"
        model_meta_path = f"{save_dir}/{model_id}_meta.json"
        if ml_config["features"].get("context_free", False): 
            with open(predictor_net_path, "wb") as f:
                pickle.dump(predictor, f)
        else:
            predictor.config_to_file(predictor_config_path)
            action_to_itemid = dict(zip(itemid_to_action.values(), itemid_to_action.keys()))
            #predictor.model_to_file(predictor_net_path)
            with open(model_meta_path,"w") as f:
                json.dump({"context_dim":context_dim,
                           "itemid_to_action":itemid_to_action,
                           "action_to_itemid":action_to_itemid,
                           "n_actions":_n_actions}, f)
        shutil.make_archive(save_dir, "zip", save_dir)

    logger.info(f"Traning took {time.time() - start} seconds.")
        
    return predictor