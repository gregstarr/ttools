import pytest
import os

from ttools import config


@pytest.fixture(autouse=True, scope='session')
def test_data_dir():
    path = os.path.join(os.path.dirname(__file__), 'ttools_data')
    config.update(path)
    yield path


@pytest.fixture
def madrigal_data_dir(test_data_dir):
    yield os.path.join(test_data_dir, 'tec_data', 'download')
