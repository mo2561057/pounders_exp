# -*- coding: utf-8 -*-
"""
Created on Aug 18

@author: Christian Zimpelmann
"""
# Scientific Computing
import numpy as np
from functools import partial

# System-specific Parameters and Functions
import sys

# Importing the module wrapping
# the criterion functions and
# the functions for the visual
# illustration. We also import
# auxiliary functions.
sys.path.insert(0, 'modules')
from src.auxiliary.simulate import simulate_sample, simulate_ols_sample

def _return_dev(endog,exog,x):
    dev = (endog - np.exp(-x[0]*exog)/(x[1] + x[2]*exog))**2
    return dev


def return_obj_func(func,endog,exog):
    out = partial(func,endog,exog)
    return out

def _return_dev_ols(endog,exog,x):
    dev = (endog - x[0] - x[1] * exog)**2
    return dev


def set_up_test_1(PARAS,START,num_agents):
    """
    This is a bit ad hoc

    """
    #Simulate values
    exog, endog = simulate_sample(num_agents, PARAS)
    # Initialize class container
    func = return_obj_func(_return_dev,endog,exog)
    return func, START



def set_up_test_2(PARAS,START,num_agents):
    """
    """
    exog, endog = simulate_ols_sample(num_agents, PARAS)
    func = return_obj_func(_return_dev_ols,endog,exog)
    return func, START


