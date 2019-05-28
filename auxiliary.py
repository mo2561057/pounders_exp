"""
This file allows for the simulation of a simple OLS relation!
"""

import os

import numpy as np
import pandas as pd


def simulate_sample(num_agents,paras):
    exog = np.random.uniform(-5,5,num_agents)
    error_term = np.random.normal(0,1,num_agents)
    endog = np.exp(-paras[0]*exog)/(paras[1]+paras[2]*exog)+error_term
    return exog,endog