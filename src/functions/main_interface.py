"""
Define the algorithm required.
"""
from petsc4py import PETSc

def solve(func,x,len_out,len_x,bounds=None):
    """
    Args:
        func:pointer to a function object that resembles the objective
        x:np.array that contains the start values of the variables of interest
        bounds: list or tuple of lists containing the bounds for the variable of interest
                The first list contains the lower value for each param and the upper list the upper value
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

    #Set the procedure for calculating the objective
    #This part has to be changed if we want more than pounders
    tao.setResidual(func_tao, crit)
    #Change they need to be in a container
    #Set the variable sounds if existing
    if bounds is not None:
        low,up = _prep_args(len(x),len(x))
        low.array = bounds[0]
        up.array=bounds[1]
        tao.setVariableBounds([low,up])

    #Set the container over which we optimize that already contians start values
    tao.setInitial(paras)

    #Run the problem
    tao.solve()

    #Create a dict that contains relevant information
    out = dict()
    out["solution"] = paras.array
    out["func_values"] = crit.array
    out["x"] = x

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




