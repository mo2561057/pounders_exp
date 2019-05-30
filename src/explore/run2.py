class OptClsLS(object):
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
        paras = PETSc.Vec().create(PETSc.COMM_WORLD)
        paras.setSizes(num_paras)

        # Create container for criterion function
        crit = PETSc.Vec().create(PETSc.COMM_WORLD)
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
            ff += dev[i]

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
        dev = (endog - paras[0]-paras[1]*exog)

        # Finishing
        return dev


#######Run first ry
# Ensure recomputability
np.random.seed(456)

# Parameterization of optimization
# problem
PARAS = [0.20, 0.12]   # True values
START = [0.10, 0.08]   # Starting values

num_agents = 10000

# wie können wir so nen SPaß simulieren ?
exog, endog = simulate_ols_sample(num_agents, PARAS)

# Initialize class container
opt_obj = OptClsLS(exog, endog, START)
func = opt_obj.form_separable_objective

# Manage PETSc objects.
paras, crit = opt_obj.create_vectors()

opt_obj.set_initial_guess(paras)

# Initialize solver instance
tao = PETSc.TAO().create(PETSc.COMM_WORLD)

tao.setType('pounders')

tao.setFromOptions()


tao.setResidual(func,R = crit)



# Solve optimization problem
tao.setInitial(paras)
tao.solve()

# Inspect solution
#plot_solution(paras, endog, exog)

# Cleanup
#paras.destroy()

#crit.destroy()

#tao.destroy()


