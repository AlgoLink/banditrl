import json
import logging
import math
from typing import Dict, NoReturn
import string
from datetime import datetime, timezone
from datetime import timedelta
from banditrl.ext.tinyflux import TinyFlux, Point

import pandas as pd

VALID_MODEL_TYPES = (
    "rliteee",
    "linucb_dict",
    "linucb_array",
    "bts",
    "lints",
    "linee",
    "logisticucb",
    "gbdt_bandit",
    "random_forest_bandit",
    "linear_bandit"
)
VALID_REWARD_TYPES = ("regression", "binary")


def read_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config


def get_logger(name: str):
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-3s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

def str_count(str):
    '''找出字符串中的中英文、空格、数字、标点符号个数'''
    count_en = count_dg = count_sp = count_zh = count_pu = 0
    str_len=0
    for s in str:
        # 英文
        if s in string.ascii_letters:
            count_en += 1
        # 数字
        elif s.isdigit():
            count_dg += 1
        # 空格
        elif s.isspace():
            count_sp += 1
        # 中文
        elif s.isalpha():
            count_zh += 1
        # 特殊字符
        else:
            count_pu += 1
            
    str_len  = count_en+count_dg+count_sp+count_pu+2*count_zh
    
    #print('英文字符：', count_en)
    #print('数字：', count_dg)
    #print('空格：', count_sp)
    #print('中文：', count_zh)
    #print('特殊字符：', count_pu)
    #print('长度：', str_len)
    return str_len
def fancy_print(text: str, color="blue", size=60):
    ansi_color = "\033[94m"
    if color == "green":
        ansi_color = "\033[95m"
    elif color == "blue":
        ansi_color = "\033[94m"
    else:
        raise Exception(f"Color {color} not supported")

    end_color = "\033[0m"
    str_len = len(text.encode())
    padding = math.ceil((size - str_len) / 2)
    header_len = padding * 2 + str_len + 2
    border = "#" * header_len
    message = "#" * padding + " " + text + " " + "#" * padding
    print(f"{ansi_color}\n{border}\n{message}\n{border}\n{end_color}")


def color_text(text: str, color="blue"):
    ansi_color = "\033[94m"
    if color == "green":
        ansi_color = "\033[95m"
    elif color == "blue":
        ansi_color = "\033[94m"
    else:
        raise Exception(f"Color {color} not supported")

    end_color = "\033[0m"
    return f"{ansi_color}{text}{end_color}"


def validate_ml_config(ml_config: Dict) -> NoReturn:
    assert "model_type" in ml_config
    assert "model_params" in ml_config
    assert "reward_type" in ml_config

    model_type = ml_config["model_type"]
    model_params = ml_config["model_params"]
    reward_type = ml_config["reward_type"]

    assert (
        model_type in VALID_MODEL_TYPES
    ), f"Model type {model_type} not supported. Valid model types are {VALID_MODEL_TYPES}"
    assert model_type in model_params
    assert (
        reward_type in VALID_REWARD_TYPES
    ), f"Reward type {reward_type} not supported. Valid reward types are {VALID_REWARD_TYPES}"


def validate_training_data_schema(training_df: pd.DataFrame) -> NoReturn:
    #TODO: add "mdp_id", "sequence_number"
    for col in ["context", "decision", "reward"]:
        assert col in training_df.columns


def pset_features_have_dense(features: Dict) -> NoReturn:
    for feature, meta in features.items():
        if meta["type"] == "P":
            if not meta["use_dense"]:
                return False
    return True

class log_data(object):
    def __init__(self,db_path):
        self.db = TinyFlux(db_path)
    def log_model_details(self,measurement,tags,fields):
        # Measurement name, a string.
        measurement = measurement
    
        # Datetime object that is "timezone-aware".
        #ts_naive = datetime.strptime(row[2], "%Y-%m-%d")
        #ts_aware = ts_naive.replace(tzinfo=ZoneInfo("US/Pacific"))
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        SHA_TZ = timezone(
            timedelta(hours=8),
            name='Asia/Shanghai',
        )
        ts_aware = utc_now.astimezone(SHA_TZ)

        # Tags as a dict of string/string key values.
        tags = tags
        # Fields as a dict of string/numeric key values.
        fields = fields
        # Initialize the Point with the above attributes.
        p = Point(
            measurement=measurement,
            time=ts_aware,
            tags=tags,
            fields=fields,
        )
        self.db.insert(p)
        return p