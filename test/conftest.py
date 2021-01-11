import pytest
import os


@pytest.fixture
def madrigal_data_dir():
    yield os.path.join(os.path.dirname(__file__), 'test_madrigal_data')


@pytest.fixture
def tec_data_dir():
    yield os.path.join(os.path.dirname(__file__), 'test_tec_data')


@pytest.fixture
def swarm_data_dir():
    yield os.path.join(os.path.dirname(__file__), 'test_swarm_data')


@pytest.fixture
def swarm_data_fn():
    yield os.path.join(os.path.dirname(__file__), 'test_swarm_data',
                       "SW_EXTD_EFIC_LP_HM_20191226T000000_20191226T235959_0102.cdf")
