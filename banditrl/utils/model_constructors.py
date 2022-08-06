from sklearn import ensemble, linear_model
from ..bandit import RliteEE,LinTS,Linucb,LinEE,LinUCB,UCB1,BTS


def build_gbdt(reward_type, learning_rate=0.1, n_estimators=100, max_depth=3):
    is_classification = reward_type == "binary"
    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    if is_classification:
        gbdt_model = ensemble.GradientBoostingClassifier(**params)
    else:
        gbdt_model = ensemble.GradientBoostingRegressor(**params)

    return gbdt_model


def build_random_forest(reward_type, n_estimators=100, max_depth=None):
    is_classification = reward_type == "binary"
    params = {"n_estimators": n_estimators, "max_depth": max_depth}

    if is_classification:
        gbdt_model = ensemble.RandomForestClassifier(**params)
    else:
        gbdt_model = ensemble.RandomForestRegressor(**params)

    return gbdt_model


def build_linear_model(reward_type, penalty="l2", alpha=1.0):
    is_classification = reward_type == "binary"
    if is_classification:
        params = {"penalty": penalty}
        linear_model_ = linear_model.LogisticRegression(**params)
    else:
        params = {"alpha": alpha}
        linear_model_ = linear_model.Ridge(**params)

    return linear_model_

def build_rliteee_model(rlite_path,model_id=None):
    return RliteEE(rlite_path,model_id)

def build_bts_model(his_context_storage,
                    model_storage,
                    alpha=1, 
                    beta = 1,
                    model_id=None):

    return BTS(history_storage=his_context_storage,
               model_storage=model_storage,
               action_storage=None,
               recommendation_cls=None,
               random_state=2022,
               action_list=[],
               alpha= alpha,
               beta = beta,
               model_id= model_id,
               campaign = None,
               init_model=False)

def build_linucb_array_model(his_context_storage,
                             model_storage,
                             action_storage,
                             n_actions,
                             context_dim,
                             alpha=0.2,
                             model_id=None):

    return Linucb(history_storage= his_context_storage,
                  model_storage= model_storage,
                  action_storage= action_storage,
                  n_actions= n_actions,
                  dim= context_dim, 
                  epsilon= alpha,
                  model_id= model_id)

def build_linucb_dict_model(his_context_storage,
                            model_storage,
                            action_storage,
                            context_dim,
                            alpha=0.2,
                            model_id=None):

    return LinUCB(history_storage=his_context_storage,
                  model_storage= model_storage,
                  action_storage = action_storage,
                  recommendation_cls= None,
                  context_dimension= context_dim, 
                  model_id= model_id,
                  alpha= alpha)