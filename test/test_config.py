import pytest
import os

from ttools import config


@pytest.fixture()
def update_config():
    base_dir = config.base_dir
    new_base_dir = "test_base_dir"
    config.update(new_base_dir)
    yield new_base_dir
    config.update(base_dir)


def test_config(update_config):
    new_base_dir = update_config
    assert config.base_dir == new_base_dir
    assert config.tec_dir == os.path.join(new_base_dir, 'tec_data')
    assert config.mlat_grid is None


def test_config_all_good():
    assert config.mlat_grid is not None
