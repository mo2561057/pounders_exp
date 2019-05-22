# -*- coding: utf-8 -*-
"""
Created on Aug 18

@author: Christian Zimpelmann
"""
#%load_ext autoreload
#%autoreload 2


# PETSC for Python
from petsc4py import PETSc
from tao4py import TAO

# Time Access and Conversions
import time

# Scientific Computing
import numpy as np

# Plotting 
import matplotlib.pyplot as plt
%pylab inline --no-import-all

# System-specific Parameters and Functions
import sys

# Importing the module wrapping
# the criterion functions and
# the functions for the visual
# illustration. We also import
# auxiliary functions.
sys.path.insert(0, 'modules')
from clsOpt import OptCls
from graphs import *
from auxiliary import *




class OptCls(object):
    """ Class to illustrate the use of the Toolkit for
        Advanced Optimization.
    """
    def __init__(self, exog, endog, START):
        """ Initialize class.
        """
        # Attach attributes
        self.exog = exog
        self.endog = endog
        self.start = START

        # Derived attributes
        self.num_agents = len(self.exog)
        self.num_paras = len(self.start)

    def create_vectors(self):
        """ Create instances of PETSc objects.
        """
        # Distribute class attributes
        num_agents = self.num_agents
        num_paras = self.num_paras

        # Create container for parameter values
        paras = PETSc.Vec().create(PETSc.COMM_SELF)
        paras.setSizes(num_paras)

        # Create container for criterion function
        crit = PETSc.Vec().create(PETSc.COMM_SELF)
        crit.setSizes(num_agents)

        # Management
        paras.setFromOptions()
        crit.setFromOptions()

        # Finishing
        return paras, crit

    def set_initial_guess(self, paras):
        """ Initialize the initial parameter values
        """
        # Set starting value
        paras[:] = self.start

    def form_separable_objective(self, tao, paras, f):
        """ Form objective function for the POUNDerS algorithm.
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Attach to PETSc object
        f.array = dev

    def form_objective(self, tao, paras):
        """ Form objective function for Nelder-Mead algorithm. The
            FOR loop results in a costly evaluation of the
            criterion function..
        """
        # Calculate deviations
        dev = self._get_deviations(paras)

        # Aggregate deviations
        ff = 0
        for i in range(self.num_agents):
            ff += dev[i]**2

        # Finishing
        return ff

    ''' Private methods
    '''
    def _get_deviations(self, paras):
        """ Get whole vector of deviations.
        """
        # Distribute class attributes
        exog = self.exog
        endog = self.endog

        # Calculate deviations
        dev = endog - np.exp(-paras[0]*exog)/(paras[1] + paras[2]*exog)

        # Finishing
        return dev

def conduct_pounders()
    # Ensure recomputability
    np.random.seed(456)

    # Parameterization of optimization 
    # problem
    PARAS = [0.20, 0.12, 0.08]   # True values
    START = [0.10, 0.08, 0.05]   # Starting values

    num_agents = 1000

    # Simulate a sample
    exog, endog = simulate_sample(num_agents, PARAS)

    # Initialize class container
    opt_obj = OptCls(exog, endog, START)

    # Manage PETSc objects.
    paras, crit = opt_obj.create_vectors()

    opt_obj.set_initial_guess(paras)

    # Initialize solver instance
    tao = TAO.Solver().create(PETSc.COMM_SELF)

    tao.setType('tao_pounders')

    tao.setFromOptions()

    tao.setSeparableObjective(opt_obj.form_separable_objective, crit)

    # Solve optimization problem
    tao.solve(paras)

    # Inspect solution
    plot_solution(paras, endog, exog)

    # Cleanup
    paras.destroy()

    crit.destroy()

    tao.destroy()

if __name__ == "__main__":
    model='baseline'
    stage='start'
    #real_inds, real_data, raw_data, sim_inds_1, \
    #sim_data_1, sim_inds_mc, sim_data_mc = compare_data()
    inds_finished, data_finished = sim_mc()
    #check_convert_raw_data()


