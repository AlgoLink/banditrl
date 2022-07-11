"""
Model storage
"""
from abc import abstractmethod
import hirlite
import pickle
import dill

class ModelStorage(object):
    """The object to store the model."""
    @abstractmethod
    def get_model(self):
        """Get model"""
        pass

    @abstractmethod
    def save_model(self):
        """Save model"""
        pass


class MemoryModelStorage(ModelStorage):
    """Store the model in memory."""
    def __init__(self):
        self._model = None

    def get_model(self):
        return self._model

    def save_model(self, model):
        self._model = model
        
class RliteModelStorage(ModelStorage):
    """Store the model in memory."""
    def __init__(self,
                 rlite_path=None,
                 model_id=None,
                 banckend='pickle'):
        if rlite_path is None:
            self.rlite_client= hirlite.Rlite("online_rlite.db",encoding='utf8')
        else:
            self.rlite_client= hirlite.Rlite(rlite_path,encoding='utf8')
        self._model_id = model_id

    def get_model(self,model_id=None):
        if model_id is None:
            model_id = self._model_id
        model_key="online:banditrl:{}".format(model_id)
        model = self.rlite_client.command("get",model_key)
        if model is not None:
            return pickle.loads(model)
        else:
            return model
        
    def save_model(self,model,model_id=None):
        if model_id is None:
            model_id = self._model_id
        model_key="online:banditrl:{}".format(model_id)
        _model = pickle.dumps(model)
        self.rlite_client.command("set",model_key,_model)
        