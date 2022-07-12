"""
History storage
"""
from abc import abstractmethod
from datetime import datetime
import hirlite
import pickle


class History(object):
    """action/reward history entry.
    Parameters
    ----------
    history_id : int
    context : {dict of list of float, None}
    recommendations : {Recommendation, list of Recommendation}
    created_at : datetime
    rewards : {float, dict of float, None}
    rewarded_at : {datetime, None}
    """

    def __init__(self,
                 history_id, 
                 context, 
                 recommendations, 
                 created_at,
                 rewarded_at=None):
        self.history_id = history_id
        self.context = context
        self.recommendations = recommendations
        self.created_at = created_at
        self.rewarded_at = rewarded_at

    def update_reward(self, rewards, rewarded_at):
        """Update reward_time and rewards.
        Parameters
        ----------
        rewards : {float, dict of float, None}
        rewarded_at : {datetime, None}
        """
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations

        for rec in recommendations:
            try:
                rec.reward = rewards[rec.action.id]
            except KeyError:
                pass
        self.rewarded_at = rewarded_at

    @property
    def rewards(self):
        if not hasattr(self.recommendations, '__iter__'):
            recommendations = (self.recommendations,)
        else:
            recommendations = self.recommendations
        rewards = {}
        for rec in recommendations:
            if rec.reward is None:
                continue
            rewards[rec.action.id] = rec.reward
        return rewards


class HistoryStorage(object):
    """The object to store the history of context, recommendations and rewards.
    """
    @abstractmethod
    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        pass

    @abstractmethod
    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        pass

    @abstractmethod
    def add_reward(self, history_id, rewards):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        pass


class MemoryHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in memory."""

    def __init__(self):
        self.histories = {}
        self.unrewarded_histories = {}
        self.n_histories = 0

    def get_history(self, history_id):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        return self.histories[history_id]

    def get_unrewarded_history(self, history_id):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        return self.unrewarded_histories[history_id]

    def add_history(self, context, recommendations, rewards=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        created_at = datetime.now()
        history_id = self.n_histories
        if rewards is None:
            history = History(history_id, context, recommendations, created_at)
            self.unrewarded_histories[history_id] = history
        else:
            rewarded_at = created_at
            history = History(history_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            self.histories[history_id] = history
        self.n_histories += 1
        return history_id

    def add_reward(self, history_id, rewards):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        rewarded_at = datetime.now()
        history = self.unrewarded_histories.pop(history_id)
        history.update_reward(rewards, rewarded_at)
        self.histories[history.history_id] = history

class RliteHistoryStorage(HistoryStorage):
    """HistoryStorage that store History objects in redis."""

    def __init__(self,rlite_path=None,model_id=None):
        if rlite_path is None:
            self.rlite_client= hirlite.Rlite("online_rlite.db",encoding='utf8')
        else:
            self.rlite_client= hirlite.Rlite(rlite_path,encoding='utf8')
        self._model_id = model_id

    def get_history(self, history_id,model_id=None):
        """Get the previous context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        if model_id is None:
            model_id = self._model_id
            
        histories_key = "banditrl:histories:{0}:{1}".format(model_id,history_id)
        histories = self.rlite_client.command("get",histories_key)
        if histories is None:
            return histories
        return pickle.loads(histories)

    def get_unrewarded_history(self, history_id,model_id=None):
        """Get the previous unrewarded context, recommendations and rewards with
        history_id.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        Returns
        -------
        history: History
        Raise
        -----
        KeyError
        """
        if model_id is None:
            model_id = self._model_id
        unrewarded_histories_key="banditrl:unrewarded_histories:{0}:{1}".format(model_id,history_id)
        unrewarded_histories = self.rlite_client.command("get",unrewarded_histories_key)
        if unrewarded_histories is None:
            return unrewarded_histories
        return pickle.loads(unrewarded_histories)

    def add_history(self, 
                    context,
                    recommendations,
                    rewards=None, 
                    request_id = None,
                    model_id=None):
        """Add a history record.
        Parameters
        ----------
        context : {dict of list of float, None}
        recommendations : {Recommendation, list of Recommendation}
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        if model_id is None:
            model_id = self._model_id
        created_at = datetime.now()
        key_n_histories="banditrl:{}".format(model_id)
        
        self.n_histories = self.rlite_client.command("get",key_n_histories)
        if self.n_histories is None:
            self.n_histories = 0
        if rewards is None:
            history = History(request_id, context, recommendations, created_at)
            _history = pickle.dumps(history)
            unrewarded_histories_key="banditrl:unrewarded_histories:{0}:{1}".format(model_id,request_id)
            self.rlite_client.command("set",unrewarded_histories_key,_history)

        else:
            rewarded_at = created_at
            history = History(request_id, context, recommendations, created_at,
                              rewards, rewarded_at)
            _history = pickle.dumps(history)
            histories_key = "banditrl:histories:{0}:{1}".format(model_id,request_id)
            self.rlite_client.command("set",histories_key,_history)
        self.n_histories = self.rlite_client.command("incr",key_n_histories)
        return request_id

    def add_reward(self, history_id, rewards,model_id=None):
        """Add reward to a history record.
        Parameters
        ----------
        history_id : int
            The history id of the history record to retrieve.
        rewards : {float, dict of float, None}
        Raise
        -----
        """
        rewarded_at = datetime.now()
        if model_id is None:
            model_id = self._model_id
        unrewarded_histories_key="banditrl:unrewarded_histories:{0}:{1}".format(model_id,history_id)
        histories_key = "banditrl:histories:{0}:{1}".format(model_id,history_id)
        unrewarded_histories = self.rlite_client.command("get",unrewarded_histories_key)
        if unrewarded_histories is not None:
            history = pickle.loads(unrewarded_histories)
            history.update_reward(rewards, rewarded_at)
            self.rlite_client.command("set",histories_key,pickle.dumps(history))
        else:
            raise KeyError("该ID尚未曝光")