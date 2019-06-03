# -*- coding: utf-8 -*-
"""
Created on Aug 18

@author: Christian Zimpelmann
"""
# Scientific Computing
import numpy as np

# System-specific Parameters and Functions
import sys

# Importing the module wrapping
# the criterion functions and
# the functions for the visual
# illustration. We also import
# auxiliary functions.
sys.path.insert(0, 'modules')
from src.auxiliary.simulate import simulate_sample, simulate_ols_sample




class _OptCls(object):
    """ Class to illustrate the use of the Toolkit for
        Advanced Optimization.
    """
    def __init__(self, exog, endog, START, num_agents):
        """ Initialize class.
        """
        # Attach attributes
        self.exog = exog
        self.endog = endog
        self.start = START
        self.num_agents = num_agents

    def form_separable_objective(self,paras):
        """ Form objective function for the POUNDerS algorithm.
        """
        # Calculate deviations
        out = self._get_deviations(paras)
        return out

    def form_separable_ols(self,paras):
        """ Form objective function for the POUNDerS algorithm.
        """
        # Calculate deviations
        out = self._get_deviations_ols(paras)
        return out

    ''' Private methods
    '''
    def _get_deviations(self, paras):
        """ Get whole vector of deviations.
        """
        # Distribute class attributes
        exog = self.exog
        endog = self.endog

        # Calculate deviations
        dev = (endog - np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog))**2

        # Finishing
        return dev

    def _get_deviations_ols(self,paras):
        exog = self.exog
        endog = self.endog

        dev = (endog - paras[0] - paras[1]*exog)**2

        return dev




def set_up_test_1(PARAS,START,num_agents):
    """
    This is a bit ad hoc

    """
    #Simulate values
    exog, endog = simulate_sample(num_agents, PARAS)
    # Initialize class container
    opt_obj = _OptCls(exog, endog, START,num_agents)
    func = opt_obj.form_separable_objective

    return func, START

def set_up_test_2(PARAS,START,num_agents,ols=False):
    """

    """
    exog, endog = simulate_ols_sample(num_agents, PARAS)
    opt_obj = _OptCls(exog, endog, START,num_agents)
    func = opt_obj.form_separable_ols
    return func, START


