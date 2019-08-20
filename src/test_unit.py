"""

"""
import numpy as np

from src.main_interface import solve
from src.simulate import simulate_ols_sample, simulate_sample
from src.create_objective import return_obj_func, _return_dev, _return_dev_ols


def test_robustness_1():
    # get random args
    PARAS = np.random.uniform(size=3)
    START = np.random.uniform(size=3)
    num_agents = 10000
    objective, x = set_up_test_1(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out), START, PARAS

    return out


def test_robustness_2():
    # get random args
    PARAS = np.random.uniform(size=2)
    START = np.random.uniform(size=2)

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out), START, PARAS

    return out


def test_box_constr():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out, bounds=bounds)
    assert 0 <= out["solution"][0] <= 0.3
    assert 0 <= out["solution"][1] <= 0.3


def test_max_iters():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]
    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out, bounds=bounds, max_iterations=25)

    assert (out["conv"] == 8 or out["conv"] == 6)
    if out["conv"] == 8:
        assert (out["sol"][0] == 25)


def test_grtol():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective,
                x,
                len_out,
                tol = {"grtol":10, "gatol":1, "gttol":1},
                bounds=bounds,
                gatol = False,
                gttol = False
                )

    assert (out["conv"] == "grtol below critical value" or out["conv"] == "step size small")

    if out["conv"] == 4:
        assert (out["sol"][2] / out["sol"][1] < 10)


def test_gatol():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective,
                x,
                len_out,
                tol={"grtol": 1, "gatol": 0.00001, "gttol": 1},
                bounds=bounds,
                grtol=False,
                gttol=False
                )
    assert (out["conv"] == "gatol below critical value" or out["conv"] == "step size small")

    if out["conv"] == 3:
        assert (out["sol"][2] < 0.00001)

def test_gttol():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective,
                x,
                len_out,
                tol={"grtol": 1, "gatol": 1, "gttol": 1},
                bounds=bounds,
                grtol=False,
                gatol=False
                )
    assert (out["conv"] == "gttol below critical value" or out["conv"] == "step size small")

    if out["conv"] == 5:
        assert (out["sol"][2] < 1)


def test_tol():
    PARAS = np.random.uniform(0.3, 0.4, size=2)
    START = np.random.uniform(0.1, 0.2, size=2)
    bounds = [[0, 0], [0.3, 0.3]]

    num_agents = 10000
    objective, x = set_up_test_2(PARAS, START, num_agents)
    len_out = len(objective(x))
    out = solve(objective, x, len_out, bounds=bounds,
                tol={"gatol": 0.00000001, "grtol": 0.00000001, "gttol": 0.0000000001})

    if out["conv"] == 3:
        assert (out["sol"][2] < 0.00000001)
    elif out["conv"] == 4:
        assert (out["sol"][2] / out["sol"][1] < 0.00000001)


def set_up_test_1(PARAS, START, num_agents):
    """
    This is a bit ad hoc

    """
    # Simulate values
    exog, endog = simulate_sample(num_agents, PARAS)
    # Initialize class container
    func = return_obj_func(_return_dev, endog, exog)
    return func, START


def set_up_test_2(PARAS, START, num_agents):
    """
    """
    exog, endog = simulate_ols_sample(num_agents, PARAS)
    func = return_obj_func(_return_dev_ols, endog, exog)
    return func, START
