"""
This file contains some functions that are important for optional arguments to the solver.
"""


def max_iters(max_iterations, tao):
    if tao.getSolutionStatus()[0] < max_iterations:
        return (0)
    elif tao.getSolutionStatus()[0] >= max_iterations:
        tao.setConvergedReason(8)

def gatol_conv(gatol,tao):
    if tao.getSolutionStatus()[2] >= gatol:
        return 0
    elif tao.getSolutionStatus()[2] < gatol:
        tao.setConvergedReason(3)

def grtol_conv(grtol, tao):
    if tao.getSolutionStatus()[2]/tao.getSolutionStatus()[1] >= grtol:
        return 0
    elif tao.getSolutionStatus()[2]/tao.getSolutionStatus()[1] < grtol:
        tao.setConvergedReason(4)



