"""
Define the algorithm required.
"""
import os

import numpy as np
from petsc4py import PETSc


def solve(func,x,constraints=None,bounds=None):
    """
    func the objective function !#
    constraints
    bounds
    initial guess
    Start with the easiest version possible!
    Only work with pounders for now.
    """
    #we want to get containers for the func verctor and the paras
    size_paras = len(x)
    size_prob = len(func(x))
    paras, crit = _prep_args(size_paras,size_prob)
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    tao.setType('pounders')
    tao.setFromOptions() #boiler plate ???
    tao.setResidual(func, crit)
    if constraints != None:
        tao.setConstraintsRoutine()
    #set up constraints
    #set up bounds
    tao.setInitial(paras)
    tao.solve()
    #Create a dict that contains relevant paras
    out = dict()
    out["solution"] = paras.array
    out["func_values"] = crit.array

    return out


def _prep_args(size_paras,size_prob):
    """
    This functions specifies relevant containers
    """
    paras = PETSc.Vec().create(PETSc.COMM_WORLD)
    paras.setSizes(num_paras)

    # Create container for criterion function
    crit = PETSc.Vec().create(PETSc.COMM_WORLD)
    crit.setSizes(num_agents)

    return paras, crit


def _set_constraints():

    pass

def _set_bounds():
    pass