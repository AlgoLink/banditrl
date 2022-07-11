"""
Action storage
"""
from abc import abstractmethod
from copy import deepcopy

import six
import hirlite
import pickle

class Action(object):
    r"""The action object
    Parameters
    ----------
    action_id: int
        The index of this action.
    """

    def __init__(self, action_id=None, action_type=None, action_text=None):
        self.id = action_id
        self.type = action_type
        self.text = action_text


class ActionStorage(object):
    """The object to store the actions."""

    @abstractmethod
    def get(self, action_id):
        r"""Get action by action id
        Parameters
        ----------
        action_id: int
            The id of the action.
        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        pass

    @abstractmethod
    def add(self, action):
        r"""Add action
        Parameters
        ----------
        action: Action object
            The Action object to add.
        Raises
        ------
        KeyError
        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        pass

    @abstractmethod
    def update(self, action):
        r"""Add action
        Parameters
        ----------
        action: Action object
            The Action object to update.
        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def remove(self, action_id):
        r"""Add action
        Parameters
        ----------
        action_id: int
            The Action id to remove.
        Raises
        ------
        KeyError
        """
        pass

    @abstractmethod
    def count(self):
        r"""Count actions
        """
        pass

    @abstractmethod
    def iterids(self):
        r"""Return iterable of the Action ids.
        Returns
        -------
        action_ids: iterable
            Action ids.
        """


class MemoryActionStorage(ActionStorage):
    """The object to store the actions using memory."""

    def __init__(self):
        self._actions = {}
        self._next_action_id = 0

    def get(self, action_id):
        r"""Get action by action id
        Parameters
        ----------
        action_id: int
            The id of the action.
        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        return deepcopy(self._actions[action_id])

    def add(self, actions):
        r"""Add actions
        Parameters
        ----------
        action: list of Action objects
            The list of Action objects to add.
        Raises
        ------
        KeyError
        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        new_action_ids = []
        for action in actions:
            if action.id is None:
                action.id = self._next_action_id
                self._next_action_id += 1
            elif action.id in self._actions:
                raise KeyError("Action id {} exists".format(action.id))
            else:
                self._next_action_id = max(self._next_action_id, action.id + 1)
            self._actions[action.id] = action
            new_action_ids.append(action.id)
        return new_action_ids

    def update(self, action):
        r"""Update action
        Parameters
        ----------
        action: Action object
            The Action object to update.
        Raises
        ------
        KeyError
        """
        self._actions[action.id] = action

    def remove(self, action_id):
        r"""Remove action
        Parameters
        ----------
        action_id: int
            The Action id to remove.
        Raises
        ------
        KeyError
        """
        del self._actions[action_id]

    def count(self):
        r"""Count actions
        Returns
        -------
        count: int
            Number of Action in the storage.
        """
        return len(self._actions)

    def iterids(self):
        r"""Return iterable of the Action ids.
        Returns
        -------
        action_ids: iterable
            Action ids.
        """
        return six.viewkeys(self._actions)

    def __iter__(self):
        return iter(six.viewvalues(self._actions))

class RliteActionStorage(ActionStorage):
    """The object to store the actions using rlite."""

    def __init__(self,rlite_path=None,model_id=None):
        if rlite_path is None:
            self.rlite_client= hirlite.Rlite("online_rlite.db",encoding='utf8')
        else:
            self.rlite_client= hirlite.Rlite(rlite_path,encoding='utf8')
        self._model_id = model_id

    def get(self, action_id,model_id=None):
        r"""Get action by action id
        Parameters
        ----------
        action_id: int
            The id of the action.
        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        if model_id is None:
            model_id = self._model_id
        model_dict_key = "_actions:dict:{}".format(model_id)

        _action = self.rlite_client.command("get",model_dict_key)
        if _action is not None:
            return pickle.loads(_action)[action_id]
        return _action


    def get_by_text(self, action_text,model_id=None):
        r"""Get action by action id
        Parameters
        ----------
        action_id: int
            The id text the action.
        Returns
        -------
        action: Action object
            The Action object that has id action_id.
        """
        if model_id is None:
            model_id = self._model_id
        _action = self.rlite_client.command("get","_actions:{0}:{1}".format(action_text,model_id))
        if _action is not None:
            return pickle.loads(_action)
        return _action

    def add(self, actions,model_id=None):
        r"""Add actions
        Parameters
        ----------
        action: list of Action objects
            The list of Action objects to add.
        Raises
        ------
        KeyError
        Returns
        -------
        new_action_ids: list of int
            The Action ids of the added Actions.
        """
        if model_id is None:
            model_id = self._model_id
        model_items_key = "_actions:onlineitems:{}".format(model_id)
        model_dict_key = "_actions:dict:{}".format(model_id)
        _actions_dict=self.rlite_client.command('get',model_dict_key)
        if _actions_dict is None :
            _actions = {}
        else:
            _actions = pickle.loads(_actions_dict)
        for action in actions:
            action_text = action.text
            action_id = action.id
            if action_text is None:
                raise KeyError("需要指定item内容：text")
            action_text_key = "_actions:{0}:{1}".format(action_text,model_id)
            _action = self.rlite_client.command("get",action_text_key)
            if _action is not None:
                raise KeyError("Action text {} exists".format(pickle.loads(_action).text))

            if action_id is None:
                action.id = self.rlite_client.command("incr","_actions:{0}:ids".format(model_id))
                action_id_key = "_actions:{0}:{1}".format(action.id,model_id)
            else:
                action_id_key = "_actions:{0}:{1}".format(action_id,model_id)
                _action_id_content = self.rlite_client.command("get",action_id_key)
                if _action_id_content is not None:
                    raise KeyError("Action id {} exists".format(pickle.loads(_action_id_content).id))

            self.rlite_client.command("set",action_id_key,pickle.dumps(action))
            self.rlite_client.command("set",action_text_key,pickle.dumps(action))
            
            self.rlite_client.command("sadd",model_items_key,str(action.id))
            _actions[action.id] = action
            _actions[action.text] = action.type

        self.rlite_client.command("set",model_dict_key,pickle.dumps(_actions))
            
        return self.rlite_client.command('SMEMBERS', model_items_key)

    def update(self, action,model_id=None):
        r"""Update action
        Parameters
        ----------
        action: Action object
            The Action object to update.
        Raises
        ------
        KeyError
        """
        if model_id is None:
            model_id = self._model_id
        model_dict_key = "_actions:dict:{}".format(model_id)
        action_text = action.text
        action_id  = action.id
        action_text_key = "_actions:{0}:{1}".format(action_text,model_id)
        action_id_key = "_actions:{0}:{1}".format(action_id,model_id)
        _actions=self.rlite_client.command('get',model_dict_key)
        if _actions is None :
            actions = {}
            actions[action.id] = action
            self.rlite_client.command("set",model_dict_key,pickle.dumps(actions))
            self.rlite_client.command("set",action_id_key,pickle.dumps(action))
            self.rlite_client.command("set",action_text_key,pickle.dumps(action))
            return True
        else:
            actions = pickle.loads(_actions)
            if action.text in actions:
                action_text_key = "_actions:{0}:{1}".format(action.text,model_id)
                _old_action = pickle.loads(self.rlite_client.command("get",action_text_key))
                del actions[action.text]
                del actions[_old_action.id]
            actions[action.id] = action
            actions[action.text] = action.type
            self.rlite_client.command("set",model_dict_key,pickle.dumps(actions))
            self.rlite_client.command("set",action_id_key,pickle.dumps(action))
            self.rlite_client.command("set",action_text_key,pickle.dumps(action))
            old_text = self.get_by_text(action.text)
            if old_text is None:
                self.rlite_client.command("set",action_text_key,pickle.dumps(action))
            else:
                _action_text_key = "_actions:{0}:{1}".format(old_text.text,model_id)
                self.rlite_client.command('del', _action_text_key)
                self.rlite_client.command("set",action_text_key,pickle.dumps(action))
                return True

    def remove(self, action_id,model_id=None):
        r"""Remove action
        Parameters
        ----------
        action_id: int
            The Action id to remove.
        Raises
        ------
        KeyError
        """
        if model_id is None:
            model_id = self._model_id
        model_dict_key = "_actions:dict:{}".format(model_id)
        _actions=self.rlite_client.command('get',model_dict_key)
        if _actions is None :
            actions = {}
            return
        else:
            actions = pickle.loads(_actions)
            _text= actions[action_id].text
            _action_text_key = "_actions:{0}:{1}".format(_text,model_id)
            action_id_key = "_actions:{0}:{1}".format(action_id,model_id)
            del actions[action_id]
            self.rlite_client.command("set",model_dict_key,pickle.dumps(actions))
            self.rlite_client.command('del', _action_text_key)
            self.rlite_client.command('del', action_id_key)
            return
            

    def count(self,model_id=None):
        r"""Count actions
        Returns
        -------
        count: int
            Number of Action in the storage.
        """
        if model_id is None:
            model_id = self._model_id
        model_items_key = "_actions:onlineitems:{}".format(model_id)
        return len(self.rlite_client.command('SMEMBERS', model_items_key))

    def iterids(self,model_id=None):
        r"""Return iterable of the Action ids.
        Returns
        -------
        action_ids: iterable
            Action ids.
        """
        if model_id is None:
            model_id = self._model_id
        model_dict_key = "_actions:dict:{}".format(model_id)
        _actions=self.rlite_client.command('get',model_dict_key)
        if _actions is None :
            actions = {}
        else:
            actions = pickle.loads(_actions)
        return six.viewkeys(actions)

    def __iter__(self,model_id=None):
        if model_id is None:
            model_id = self._model_id
        model_dict_key = "_actions:dict:{}".format(model_id)
        _actions=self.rlite_client.command('get',model_dict_key)
        if _actions is None :
            actions = {}
        else:
            actions = pickle.loads(_actions)
        return iter(six.viewvalues(actions))
    
    @property
    def actions(self):
        """Get actions dict."""
        model_dict_key = "_actions:dict:{}".format(self._model_id)
        _actions=self.rlite_client.command('get',model_dict_key)
        if _actions is None :
            actions = {}
        else:
            actions = pickle.loads(_actions)
        return actions