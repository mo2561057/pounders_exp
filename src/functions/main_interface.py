"""
Define the algorithm required.
"""
import os

import numpy as np
from petsc4py import PETSc


def solve(func,x,bounds=None):
    """
    Args:
        func:pointer to a function object that resembles the objective
        x:list that contains the start values of the variables of interest
        bounds: list or tuple of lists containing the bounds for the variable of interest
    Returns:
        out: dict containing the solution param and the optimal values of the objective
    """
    #we want to get containers for the func verctor and the paras
    size_paras = len(x)
    size_objective = len(func(x))
    paras, crit = _prep_args(size_paras,size_objective)

    #Set the start value
    paras[:] = x

    #Create the solver object
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)
    #Set the solver type
    tao.setType('pounders')

    #Set the procedure for calculating the objective
    #This part has to be changed if we want more than pounders
    tao.setResidual(func, crit)

    #Set the variable sounds if existing
    if bounds != None:
        tao.setVarBounds(bounds)

    #Set the container over which we optimize that already contians start values
    tao.setInitial(paras)

    #Run the problem
    tao.solve()

    #Create a dict that contains relevant information
    out = dict()
    out["solution"] = paras.array
    out["func_values"] = crit.array

    #Destroy petsc objects for memory reasons
    tao.destroy()
    paras.destroy()
    crit.destroy()

    return out


def _prep_args(size_paras,size_objective):
    """
    Args:
        size_paras: int containing the size of the pram vector
        size_prob: int containing the size of the
    """
    #create container for variable of interest
    paras = PETSc.Vec().create(PETSc.COMM_WORLD)
    paras.setSizes(size_paras)

    # Create container for criterion function
    crit = PETSc.Vec().create(PETSc.COMM_WORLD)
    crit.setSizes(size_objective)

    #Initialize
    crit.setFromOptions()
    paras.setFromOptions()

    return paras, crit




