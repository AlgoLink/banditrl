## 模型训练

```
from banditrl.training.trainer import train
import pandas as pd

df = pd.read_csv("height_dataset.csv")

ml_config = {
    "model_id": "model_rliteee_v1",
    "storage":{
        "model":{"type":"rlite",
                 "path":"model.db"},
        "his_context":{},
        "action":{},
        "predictor_save_dir":None
    },
    "features":{"context_free":True},
    "model_type": "rliteee",
    "reward_type": "regression",
    "model_params": {"rliteee":{}}
}

test = train(
    training_df= df,
    ml_config= ml_config,
    feature_config= None,
    itemid_to_action ={},
    offline_train = True
)
```
训练日志
![train logs](../resources/rliteee_train.jpg)
