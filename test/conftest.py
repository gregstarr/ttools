import pytest
import os


@pytest.fixture()
def test_data_dir():
    yield os.path.join(os.path.dirname(__file__), 'ttools_data')


@pytest.fixture
def madrigal_data_dir(test_data_dir):
    yield os.path.join(test_data_dir, 'madrigal')


@pytest.fixture
def tec_data_dir(test_data_dir):
    yield os.path.join(test_data_dir, 'tec_data')


@pytest.fixture
def swarm_data_dir(test_data_dir):
    yield os.path.join(test_data_dir, 'swarm')


@pytest.fixture
def swarm_data_dir(test_data_dir):
    yield os.path.join(test_data_dir, 'swarm')