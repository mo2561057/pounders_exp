"""
Define the algorithm required.
"""
from petsc4py import PETSc
from functools import partial

from src.auxiliary import max_iters, gatol_conv, grtol_conv

def solve(func,
          x,
          len_out,
          len_x,
          bounds=None,
          init_tr = None,
          tol = {"gatol":0.00000001,"grtol":0.00000001,"gttol":0.0000000001},
          max_iterations = None,
          gatol = None,
          grtol = None
          ):
    """
    Args:
        func:pointer to a function object that resembles the objective
        x:np.array that contains the start values of the variables of interest
        bounds: list or tuple of lists containing the bounds for the variable of interest
                The first list contains the lower value for each param and the upper list the upper value
        init_tr: Sets the radius for the initil trust region
        tol: Sets the tolarance for the three default stopping criteria. The routine will stop once the first is reached
        max_iterations: ALternative Stopping criterion. If set the routine will stop after the number of specified
                        iterations or after the step size is sufficiently small.
        gatol: This allows to set an explicit stopping criterion for the norm of the approximated gradient. The routine
               will either stop when the gradient satisfies the condition or when the step size is sufficiently small.
        grtol: This allows to set an explicit stopping criterion for the norm of the approximated gradient divided by
                the function value .The routine will either stop when the value satisfies the condition or when the
                git cstep size is sufficiently small
    Returns:                                                                        
        out: dict containing the solution param and the optimal values of the objective
    """
    #we want to get containers for the func verctor and the paras
    size_paras = len_x
    size_objective = len_out
    paras, crit = _prep_args(size_paras,size_objective)

    #Set the start value
    paras[:] = x
    def func_tao(tao,paras,f):
        dev = func(paras.array)
        # Attach to PETSc object
        f.array = dev

    #Create the solver object
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)


    #Set the solver type
    tao.setType('pounders')
    #tao.setConvergenceTest(2)

    tao.setFromOptions()

    #Set the procedure for calculating the objective
    #This part has to be changed if we want more than pounders
    tao.setResidual(func_tao, crit)



    #We try to set user defined convergence tests
    if init_tr is not None:
        tao.setInitialTrustRegionRadius(init_tr)


    #Change they need to be in a container
    #Set the variable sounds if existing
    if bounds is not None:
        low,up = _prep_args(len(x),len(x))
        low.array = bounds[0]
        up.array=bounds[1]
        tao.setVariableBounds([low,up])

    #Set the container over which we optimize that already contians start values
    tao.setInitial(paras)

    #Set tolerances for default convergence tests
    tao.setTolerances(gatol=tol["gatol"],gttol=tol["gttol"],grtol=tol["grtol"])

    #Set user defined convergence tests. Beware that specifiying multiple tests could overwrite others or lead to
    # unclear behavior.
    if max_iterations is not None:
        tao.setConvergenceTest(partial(max_iters,max_iterations))
    if grtol is not None:
        tao.setConvergenceTest(partial(grtol_conv,grtol))
    if gatol is not None:
        tao.setConvergenceTest(partial(gatol_conv,gatol))

    #Run the problem
    tao.solve()

    #Create a dict that contains relevant information
    out = dict()
    out["solution"] = paras.array
    out["func_values"] = crit.array
    out["x"] = x
    out["conv"] = tao.getConvergedReason()
    out["sol"] = tao.getSolutionStatus()
    out["tol"] = tao.getTolerances()

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




