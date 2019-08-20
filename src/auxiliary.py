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

def grtol_gatol_conv(grtol, gatol, tao):
    if tao.getSolutionStatus()[2]/tao.getSolutionStatus()[1] >= grtol:
        return 0
    elif tao.getSolutionStatus()[2]/tao.getSolutionStatus()[1] < grtol:
        tao.setConvergedReason(4)

    elif tao.getSolutionStatus()[2] < gatol:
        tao.setConvergedReason(3)

def get_tolerances(tol, gatol, grtol):
    out = tol.copy()
    if gatol is False and grtol is False:
        out["gatol"] = -1
        out["grtol"] = -1
    elif gatol is False:
        out["gatol"] = -1
    elif grtol is False:
        out["grtol"] = -1
    return out


conv_reason = {3: "gatol below critical value",
               4: "grtol below critical value",
               5: "gttol below critical value",
               6: "step size small",
               7: "objective below min value",
               8: "user defined",
               -2: "maxits reached",
               -4: "numerical porblems",
               -5: "max funcevals reached",
               -6: "line search failure",
               -7: "trust region failure",
               -8: "user defined"
               }



