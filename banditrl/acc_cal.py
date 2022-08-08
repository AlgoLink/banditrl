import json
from banditrl.utils import utils

logger = utils.get_logger(__name__)

def cal_acc(policy,df,topN=1,model_id=None,data_from='Testing'):
    """Calculate cumulative acc with respect to time.
        Parameters
        ----------
        policy: bandit object
            The bandit algorithm you want to evaluate.
        Return
        ---------
        cum_reward: int
            The value is accuracy .
        cum_n_actions: int
            cumulative number of recommended actions}.
    """

    cum_reward = 0
    cum_n_actions = 0
    for index,rows in df.iterrows():
        _features = rows["context"]
        decision = rows["decision"]
        try:
            features=json.loads(_features)
        except:
            features = _features
            
        #print(features,type(features))
        request_id = "{}_{}".format(index,model_id)
        recom = policy.get_action(features,request_id,model_id,topN)
        cum_n_actions+=1
        if recom[0]==decision:
            cum_reward+=1

    avg_reward = cum_reward/cum_n_actions
    logger.info(utils.color_text(f"all acc predictor:{cum_reward}", color="blue"))
    logger.info(utils.color_text(f"all actions:{cum_n_actions}", color="blue"))
    logger.info(utils.color_text(f"{data_from} accuracy: {avg_reward}", color="blue"))

    return avg_reward
#cal_acc(test,df,1,"model_logistic_v1")