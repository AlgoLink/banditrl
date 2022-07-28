from banditrl.preprocessing import preprocessor
import pandas as pd
import numpy as np
import torch
import json
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from collections import defaultdict
import pickle


class BanditContextEncoder:
    """Class used to make features given for bandit model."""
    def __init__(
        self,
        feature_config,
        float_feature_order,
        id_feature_order,
        id_feature_str_to_int_map,
        transforms,
        imputers,
        model,
        model_type,
        reward_type,
        model_spec=None,
        dense_features_to_use=["*"],
    ):
        self.feature_config = feature_config
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.id_feature_str_to_int_map = id_feature_str_to_int_map
        self.transforms = transforms
        self.imputers = imputers
        self.model = model
        self.model_type = model_type
        self.reward_type = reward_type
        self.model_spec = model_spec
        self.dense_features_to_use = dense_features_to_use

        # the ordered choices that we need to score over.
        self.choices = feature_config["choices"]

        # if using dropout to get prediction uncertainty, how many times to score
        # the same observation
        self.num_times_to_score = 30
        self.ucb_percentile = 90

    def transform_feature(self, vals, transformer=None, imputer=None):
        vals = vals.reshape(-1, 1)
        if imputer:
            vals = imputer.transform(vals)
        if transformer:
            vals = transformer.transform(vals)

        return vals

    def preprocess_input(self, input, choices):
        # score input choices or all choices by default if none provided.
        # expand the input across all of these choices
        self.scored_choices = choices = choices or self.choices
        expanded_input = [
            dict(input, **{preprocessor.DECISION_FEATURE_NAME: d})
            for d in self.scored_choices
        ]

        df = pd.DataFrame(expanded_input)
        float_feature_array = np.empty((len(df), 0))
        id_list_feature_array = np.empty((len(df), 0))

        for feature_name in self.float_feature_order:
            if feature_name not in df.columns:

                if feature_name == preprocessor.POSITION_FEATURE_NAME:
                    # always score position feature at a fixed position
                    # since including it in training was just meant to debias
                    df[feature_name] = 0

                else:
                    # context is missing this feature, that is fine
                    logger.warning(
                        f"'{feature_name}' expected in context, but missing."
                    )
                    if self.feature_config["features"][feature_name]["type"] == "C":
                        df[feature_name] = preprocessor.MISSING_CATEGORICAL_CATEGORY
                    else:
                        df[feature_name] = None

            values = self.transform_feature(
                df[feature_name].values,
                self.transforms[feature_name],
                self.imputers[feature_name],
            )
            float_feature_array = np.append(float_feature_array, values, axis=1)

        for feature_name in self.id_feature_order:
            meta = self.feature_config["features"][feature_name]
            product_set_id = meta["product_set_id"]
            product_set_meta = self.feature_config["product_sets"][product_set_id]

            if feature_name not in df.columns:
                # Handle passing missing product set feature
                logger.warning(f"'{feature_name}' expected in context, but missing.")
                df[feature_name] = None

            if meta["use_dense"] is True and "dense" in product_set_meta:

                dense = defaultdict(list)
                # TODO: don't like that this is O(n^2), think about better way to do this
                for val in df[feature_name].values:
                    if not isinstance(val, list):
                        val = [val]

                    dense_matrix = []
                    for v in val:
                        dense_features = product_set_meta["dense"].get(v)
                        if not dense_features:
                            logger.warning(
                                f"No dense representation found for '{feature_name}'"
                                f" product set value '{v}'."
                            )
                        else:
                            dense_matrix.append(dense_features)

                    if not dense_matrix:
                        # there were no or no valid id features to add, add an
                        # empty row to be imputed
                        dense_matrix.append([])

                    for idx, feature_spec in enumerate(product_set_meta["features"]):
                        dense_feature_name = feature_spec["name"]
                        row_vals = []
                        for row in dense_matrix:
                            if not row:
                                dense_feature_val = (
                                    preprocessor.MISSING_CATEGORICAL_CATEGORY
                                    if feature_spec["type"] == "C"
                                    else None
                                )
                            else:
                                dense_feature_val = row[idx]
                            row_vals.append(dense_feature_val)

                        dense_feature_val = preprocessor.flatten_dense_id_list_feature(
                            row_vals, feature_spec["type"]
                        )
                        dense[dense_feature_name].append(dense_feature_val)

                for idx, feature_spec in enumerate(product_set_meta["features"]):
                    dense_feature_name_desc = f"{feature_name}:{feature_spec['name']}"
                    if (
                        self.dense_features_to_use != ["*"]
                        and dense_feature_name_desc not in self.dense_features_to_use
                    ):
                        continue

                    dtype = (
                        np.dtype(float)
                        if feature_spec["type"] == "N"
                        else np.dtype(object)
                    )

                    vals = dense[feature_spec["name"]]
                    if feature_spec["type"] == "C":
                        # fill in null categorical values with a "null" category
                        vals = [
                            preprocessor.MISSING_CATEGORICAL_CATEGORY
                            if v is None
                            else v
                            for v in vals
                        ]

                    vals = np.array(vals, dtype=dtype)
                    values = self.transform_feature(
                        vals,
                        self.transforms[dense_feature_name_desc],
                        self.imputers[dense_feature_name_desc],
                    )
                    float_feature_array = np.append(float_feature_array, values, axis=1)
            else:
                # sparse id list features need to be converted from string to int,
                # but aside from that are not imputed or transformed.
                str_to_int_map = self.id_feature_str_to_int_map[product_set_id]
                # if the feature value is not present in the map, assign it to 0
                # which corresponds to the null embedding row
                values = self.transform_feature(
                    df[feature_name].apply(lambda x: str_to_int_map.get(x, 0)).values
                )
                id_list_feature_array = np.append(id_list_feature_array, values, axis=1)

        return {
            "X_float": pd.DataFrame(float_feature_array),
            "X_id_list": pd.DataFrame(id_list_feature_array),
        }

    def preprocessed_input_to_pytorch(self, data):
        X, _ = preprocessor.data_to_pytorch(data)
        return X

    def predict(self, input, choices=None, get_ucb_scores=False):
        """
        If `get_ucb_scores` is True, get upper confidence bound scores which
        requires a model trained with dropout and for the model to be in train()
        mode (eval model turns off dropout by default).
        """
        input = self.preprocess_input(input, choices)
        pytorch_input = self.preprocessed_input_to_pytorch(input)

        

        return {
            #"input": input,
            "pytorch_input": pytorch_input
        }


    def config_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method writes
        the parameters to reconstruct the preprocessing & data imputation
        objects/logic to a JSON file.
        """
        output = {
            "model_type": self.model_type,
            "model_spec": self.model_spec,
            "feature_config": self.feature_config,
            "float_feature_order": self.float_feature_order,
            "id_feature_order": self.id_feature_order,
            "id_feature_str_to_int_map": self.id_feature_str_to_int_map,
            "dense_features_to_use": self.dense_features_to_use,
            "reward_type": self.reward_type,
            "transforms": {},
            "imputers": {},
        }

        # write the parameters of the feature transformers
        for feature_name, transform in self.transforms.items():
            if transform is None:
                # id lists don't have transforms
                spec = None
            elif isinstance(transform, preprocessing.StandardScaler):
                spec = {
                    "name": "StandardScaler",
                    "mean": transform.mean_.tolist(),
                    "var": transform.var_.tolist(),
                    "scale": transform.scale_.tolist(),
                }
            elif isinstance(transform, preprocessing.OneHotEncoder):
                spec = {
                    "name": "OneHotEncoder",
                    "obj":transform,
                    "categories": [transform.categories_[0].tolist()],
                    "sparse": transform.sparse,
                }
            else:
                raise Exception(
                    f"Don't know how to serialize preprocessor of type {type(transform)}"
                )
            output["transforms"][feature_name] = spec

        # write the parameters of the feature imputers
        for feature_name, imputer in self.imputers.items():
            if imputer is None:
                # categorical & id lists don't have imputers
                spec = None
            else:
                spec = {
                    "parameters": imputer.get_params(),
                    "statistics": imputer.statistics_.tolist(),
                }
            output["imputers"][feature_name] = spec

        with open(path, "wb") as f:
            pickle.dump(output, f)

    @staticmethod
    def encoder_from_file(config_path):
        with open(config_path, "rb") as f:
            config_dict = pickle.load(f)

        # initialize transforms
        transforms = {}
        for feature_name, transform_spec in config_dict["transforms"].items():
            if transform_spec is None:
                # id lists don't have transforms
                transform = None
            elif transform_spec["name"] == "StandardScaler":
                transform = preprocessing.StandardScaler()
                transform.mean_ = np.array(transform_spec["mean"])
                transform.scale_ = np.array(transform_spec["scale"])
                transform.var_ = np.array(transform_spec["var"])
            elif transform_spec["name"] == "OneHotEncoder":
                transform = transform_spec["obj"]#preprocessing.OneHotEncoder()
                #transform.sparse = transform_spec["sparse"]
                #transform.categories_ = np.array(transform_spec["categories"])
            else:
                raise Exception(
                    f"Don't know how to load transform_spec of type {transform_spec['name']}"
                )
            transforms[feature_name] = transform

        # initialize imputers
        imputers = {}
        for feature_name, imputer_spec in config_dict["imputers"].items():
            if imputer_spec is None:
                # categoricals & id lists don't have imputers
                imputer = None
            else:
                imputer = SimpleImputer()
                imputer.set_params(**imputer_spec["parameters"])
                imputer.statistics_ = np.array(imputer_spec["statistics"])
            imputers[feature_name] = imputer

        return BanditContextEncoder(
            feature_config=config_dict["feature_config"],
            float_feature_order=config_dict["float_feature_order"],
            id_feature_order=config_dict["id_feature_order"],
            id_feature_str_to_int_map=config_dict["id_feature_str_to_int_map"],
            transforms=transforms,
            imputers=imputers,
            model="model",
            model_type=config_dict["model_type"],
            reward_type=config_dict["reward_type"],
            model_spec=config_dict["model_spec"],
            dense_features_to_use=config_dict["dense_features_to_use"],
        )
