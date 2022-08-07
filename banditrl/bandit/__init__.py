"""Bandit algorithm classes
"""

from .exp3 import Exp3
from .exp4p import Exp4P
from .linthompsamp import LinThompSamp
from .linucb import LinUCB
from .ucb1 import UCB1
from .usermodel import RliteEE,BTS
from .linear import LinUCB as Linucb
from .linear import LinTS
from .linear import LinEpsilonGreedy as LinEE
from .logistic import LogisticUCB

__all__ = [
    'Exp3', 
    'Exp4P', 
    'LinThompSamp', 
    'LinUCB', 
    'UCB1',
    'RliteEE',
    'BTS',
    'Linucb',
    'LinEE',
    'LinTS',
    'LogisticUCB']