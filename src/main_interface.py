"""
Define the algorithm required.
"""
from petsc4py import PETSc
from functools import partial

from src.auxiliary import max_iters, gatol_conv, grtol_conv, grtol_gatol_conv, get_tolerances, conv_reason


def solve(func,
          x,
          len_out,
          bounds=None,
          init_tr=None,
          tol={"gatol": 0.00000001, "grtol": 0.00000001, "gttol": 0.0000000001},
          max_iterations=None,
          gatol=True,
          grtol=True,
          gttol=True
          ):
    """
    Args:
        func: function that takes a 1d numpy array and returns a 1d numpy array
        x:np.array that contains the start values of the variables of interest
        bounds: list or tuple of lists containing the bounds for the variable of interest
                The first list contains the lower value for each param and the upper list the upper value
        init_tr: Sets the radius for the initial trust region that the optimizer employs. 
        tol: Sets the tolerance for the three default stopping criteria. The routine will stop once the first is reached.
             One can turn off specific criteria with other args. In this case their value in this dict does not matter.
        max_iterations: Alternative Stopping criterion. If set the routine will stop after the number of specified
                        iterations or after the step size is sufficiently small. If the variable is set the default
                        criteria will all be ignored.
        gatol: Boolean that indicates whether the gatol should be cosnidered.
               Explicit description is in the documentation.
        grtol: Boolean that indicates whether the grtol should be cosnidered.
               Explicit description is in the documentation
        gttol: Boolean that indicates whether the gttol should be cosnidered.
               Explicit description is in the documentation
    Returns:                                                                        
        out: dict with the following key value pairs:
             "solution": solution vector as np.array,
             "func values": np.array of value of the objective at the solution
             "x": np.array of the start values
             "conv": string indicating the termination reason
             "sol": list containing: current iterate as int, current value of the objective as float, current value of
                    the approximated jacobian as float, infeasability norm as float, step length as float and termination
                    reason as int.


    """
    # we want to get containers for the func verctor and the paras
    size_paras = len(x)
    size_objective = len_out
    paras, crit = _prep_args(size_paras, size_objective)

    # Set the start value
    paras[:] = x

    def func_tao(tao, paras, f):
        """
        This function takes an input, calculates the value of the objective and
        attaches it to an petsc object f thereafter.
        func_tao puts the objective in a format that the optimizer requires.
        Args:
             tao: The tao object we created for the optimization task
             paras: 1d np.array of the current values at which we want to evaluate the function.
             f: Petsc object in which we save the current function value
        """
        dev = func(paras.array)
        # Attach to PETSc object
        f.array = dev

    # Create the solver object
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type
    tao.setType('pounders')

    tao.setFromOptions()

    # Set the procedure for calculating the objective
    # This part has to be changed if we want more than pounders
    tao.setResidual(func_tao, crit)

    # We try to set user defined convergence tests
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)

    # Change they need to be in a container
    # Set the variable sounds if existing
    if bounds is not None:
        low, up = _prep_args(len(x), len(x))
        low.array = bounds[0]
        up.array = bounds[1]
        tao.setVariableBounds([low, up])

    # Set the container over which we optimize that already contians start values
    tao.setInitial(paras)

    # Obtain tolerances for the convergence criteria
    # Since we can not create gttol manually we manually set gatol and or grtol to zero once a subset of these two is
    # turned off and gttol is still turned on
    tol_real = get_tolerances(tol, gatol, grtol)

    # Set tolerances for default convergence tests
    tao.setTolerances(gatol=tol_real["gatol"], gttol=tol_real["gttol"], grtol=tol_real["grtol"])

    # Set user defined convergence tests. Beware that specifiying multiple tests could overwrite others or lead to
    # unclear behavior.
    if max_iterations is not None:
        tao.setConvergenceTest(partial(max_iters, max_iterations))
    elif gttol is False and gatol is False:
        tao.setConvergenceTest(partial(grtol_conv, tol["grtol"]))
    elif gatol is False and gttol is False:
        tao.setConvergenceTest(partial(gatol_conv, tol["gatol"]))
    elif gttol is False:
        tao.setConvergenceTest(partial(grtol_gatol_conv, tol["grtol"], tol["gatol"]))

    # Run the problem
    tao.solve()

    # Create a dict that contains relevant information
    out = dict()
    out["solution"] = paras.array
    out["func_values"] = crit.array
    out["x"] = x
    out["conv"] = conv_reason[tao.getConvergedReason()]
    out["sol"] = tao.getSolutionStatus()

    # Destroy petsc objects for memory reasons
    tao.destroy()
    paras.destroy()
    crit.destroy()

    return out


def _prep_args(size_paras, size_objective):
    """
    Args:
        size_paras: int containing the size of the pram vector
        size_prob: int containing the size of the
    """
    # create container for variable of interest
    paras = PETSc.Vec().create(PETSc.COMM_WORLD)
    paras.setSizes(size_paras)

    # Create container for criterion function
    crit = PETSc.Vec().create(PETSc.COMM_WORLD)
    crit.setSizes(size_objective)

    # Initialize
    crit.setFromOptions()
    paras.setFromOptions()

    return paras, crit
