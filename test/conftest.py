import pytest
import os
import cvxpy as cp

from ttools import config


@pytest.fixture(autouse=True, scope='session')
def setup():
    path = os.path.join(os.path.dirname(__file__), 'ttools_data')
    config.update(path)
    if 'GUROBI' not in cp.installed_solvers():
        config.SOLVER = cp.ECOS
    yield path


@pytest.fixture
def madrigal_data_dir(setup):
    yield os.path.join(setup, 'tec_data', 'download')
