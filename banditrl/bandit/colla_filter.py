"""
Persistant storage for players, using Rlite.
"""
import hirlite
import pickle

#from . import config
def int_or_none(value):
    if value is None:
        return value
    return int(value)


def set_to_ints(values):
    """
    Redis returns all values as strings. This optimistically converts
    a set of strings to a set of integers.
    """
    ints = []
    for value in values:
        ints.append(int(value))
    return set(ints)



class RliteBackend(object):
    """
    Movie database storage and access functions, backed by Redis. Currently,
    only 'create' and 'retrieve' type functions are implemented.
    """

    def __init__(self,rlite_path=None,model_id=None):
        if rlite_path is None:
            self.rlite_client= hirlite.Rlite("online_rlite.db",encoding='utf8')
        else:
            self.rlite_client= hirlite.Rlite(rlite_path,encoding='utf8')
        self._model_id = model_id

    # 设置每个用户感兴趣的集合。
    def add_user_interested_cat(self,uid,category,model_id=None):
        if model_id is None:
            model_id = self._model_id
        cat_key="model:{0}:user:{1}:categories".format(model_id,uid)
        self.rlite_client.command("SADD",cat_key,category)
        return True
    # 为每个类别保持一个集合，使该集合包含该类别的所有项目。
    def add_item(self,category,item,model_id=None):
        if model_id is None:
            model_id = self._model_id
            
        item_key = "model:{0}:category:{1}:items".format(model_id,category)
        self.rlite_client.command("SADD",item_key,item)
        return True
    # 获取用户感兴趣的所有类别（假设这是一个小集合。 对于大数据集，使用SSCAN）。
    def get_user_cats(self,uid,model_id=None):
        if model_id is None:
            model_id = self._model_id
        cat_key="model:{0}:user:{1}:categories".format(model_id,uid)
        return self.rlite_client.command("SMEMBERS",cat_key)
    
    # 获取所有属于用户感兴趣的类别的项目。SUNIONSTORE
    def get_cat_items(self,cat_list,model_id=None):
        if model_id is None:
            model_id = self._model_id
        key_uion = ''
        for cat in cat_list:
            ki="model:{0}:category:{1}:items|".format(model_id,cat)
            key_uion+=ki
        unikey = key_uion[0:len(key_uion)-1].split("|")
        
        return self.rlite_client.command("SUNION",*unikey)
    #Collaborative Filtering Based on User-Item Associations
    #维护一个与用户相关的所有项目的集合，例如，通过电子商务应用购买的所有项目。
    def add_user_item(self,uid,item,model_id=None):
        if model_id is None:
            model_id = self._model_id
        user_item_key="model:{0}:user:{1}:items".format(model_id,uid)
        self.rlite_client.command("SADD",user_item_key, item)

        return True
    # 对于每个用户到项目的关联，维护一个项目到用户的反向映射。
    def add_item_to_user(self,count,item, uid, model_id=None):
        if model_id is None:
            model_id = self._model_id
        item_key="model:{0}:item:{1}:users".format(model_id,item)
        self.rlite_client.command("SADD",item_key,count, uid)
        
        return True
    # 获取所有与用户相关的项目（假设这是一个小的集合。 对于一个大的数据集，使用SSCAN）
    def get_user_items(self,uid,model_id=None):
        if model_id is None:
            model_id = self._model_id
        user_key = "model:{0}:user:{1}:items".format(model_id,uid)
        
        return self.rlite_client.command("SMEMBERS",user_key)
    # 获取所有属于用户感兴趣的类别的用户。
    def get_item_users(self, item_list,model_id=None):
        if model_id is None:
            model_id = self._model_id
        key_uion = ''
        for item in item_list:
            ki="model:{0}:item:{1}:users|".format(model_id,item)
            key_uion+=ki
        unikey = key_uion[0:len(key_uion)-1].split("|")
        
        return self.rlite_client.command("SUNION",*unikey)

    # 获取所有属于用户感兴趣的类别的项目。
    # 下面计算的最终集合将包含所有与其他用户有相同项目关联的项目。
    def get_user_all_recom_items(self, uid, uid_list,model_id=None):
        if model_id is None:
            model_id = self._model_id
            
        key_user = "model:{0}:user:{1}:all_recommended".format(model_id,uid)
        key_uion = ''
        for _uid in uid_list:
            ki="model:{0}:user:{1}:items|".format(model_id,_uid)
            key_uion+=ki
        unikey = key_uion[0:len(key_uion)-1].split("|")
        
        return self.rlite_client.command("SUNIONSTORE",key_user,*unikey)
    # 获取尚未与该用户关联，但与其他有类似行为的用户关联的项目列表。
    def get_user_unrecom_items(self,uid,model_id=None):
        if model_id is None:
            model_id = self._model_id
        k1 = "model:{0}:user:{1}:all_recommended".format(model_id,uid)
        k2 = "model:{0}:user:{1}:items".format(model_id,uid)
        return self.rlite_client.command("SDIFF",k1,k2)
    
    # Collaborative Filtering Based on User-Item Associations and Their Ratings
    # 为每个用户维护一个排序集，以存储该用户评价的所有项目。
    def add_user_item_rating(self,uid,item,rating,model_id=None):
        if model_id is None:
            model_id = self._model_id
        ukey="model:{0}:user:{1}:items".format(model_id,uid)
        _rating = "-{}".format(rating)
        self.rlite_client.command("ZADD",ukey,_rating, item)
        return True
    # Step 1 Insert rating events
    # 为每个项目拥有一个排序集；跟踪所有为该项目评分的用户。
    def add_uid_item_scores(self,uid,item,rating,model_id=None):
        if model_id is None:
            model_id = self._model_id
        ukey="model:{0}:item:{1}:scores".format(model_id,item)
        _rating = "-{}".format(rating)
        self.rlite_client.command("ZADD",ukey,_rating, uid)
        return True
    # Step 2 Get candidates with the same item ratings
    def get_user_cand(self,uid,topN=1, model_id=None):
        if model_id is None:
            model_id = self._model_id
        ukey="model:{0}:user:{1}:items".format(model_id,uid)
        return self.rlite_client.command("ZRANGE",ukey, "0" ,str(topN))
    # Find users who have rated the same items
    def get_user_rated_same_items(self, uid, item_list,model_id=None):
        if model_id is None:
            model_id = self._model_id
            
        key_user = "model:{0}:user:{1}:same_items".format(model_id,uid)
        key_uion = ''
        for item in item_list:
            ki="model:{0}:item:{1}:scores|".format(model_id,item)
            key_uion+=ki
        unikey = key_uion[0:len(key_uion)-1].split("|")
        
        return self.rlite_client.command("SUNIONSTORE",key_user,*unikey)

    # Step 3 Calculate similarity for each candidate
    def get_sim_cand(self,uid,topN=1,model_id=None):
        if model_id is None:
            model_id = self._model_id
        ukey = "model:{0}:user:{1}:same_items".format(model_id,uid)
        
        return self.rlite_client.command("ZRANGE",ukey, "0" ,str(topN))