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
    len_x = len(x)
    len_out = len(objective(x))
    out = solve(objective,x,len_out,len_x),START,PARAS

    return out

def test_robustness_2():
    #get random args
    PARAS = np.random.uniform(size=2)
    START = np.random.uniform(size=2)


    num_agents = 10000
    objective,x = set_up_test_2(PARAS,START,num_agents)
    len_x = len(x)
    len_out = len(objective(x))
    out = solve(objective,x,len_out,len_x),START,PARAS

    return out

def test_box_constr():
    PARAS = np.random.uniform(0.3,0.4,size=2)
    START = np.random.uniform(0.1,0.2,size=2)
    bounds = [[0,0],[0.3,0.3]]

    num_agents = 10000
    objective,x = set_up_test_2(PARAS,START,num_agents)
    len_x = len(x)
    len_out = len(objective(x))
    out = solve(objective,x,len_out,len_x,bounds = bounds)
    assert 0 <= out["solution"][0] <= 0.3
    assert 0 <= out["solution"][1] <= 0.3


