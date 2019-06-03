"""

"""
import numpy as np

from src.functions.main_interface import solve
from src.auxiliary.create_objective import set_up_test_1,set_up_test_2


def test_robustness_1():
    #get random args
    PARAS = np.random.uniform(size=3)
    START = np.random.uniform(size=3)
    num_agents = np.random.randint(1000)
    objective,x = set_up_test_1(PARAS,START,num_agents)
    out = solve(objective,x),START,PARAS

    return out, PARAS, START

def test_robustness_2():
    #get random args
    PARAS = np.random.uniform(size=2)
    START = np.random.uniform(size=2)


    num_agents = 10000
    objective,x = set_up_test_2(PARAS,START,num_agents)
    out = solve(objective,x),START,PARAS

    return out, PARAS, START