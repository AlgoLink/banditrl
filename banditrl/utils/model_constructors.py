from sklearn import ensemble, linear_model

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