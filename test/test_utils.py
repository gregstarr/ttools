import numpy as np
import pytest

from ttools import utils


@pytest.fixture()
def dt64():
    yield np.datetime64('2000-01-01T00:00:00') + np.arange(10) * np.timedelta64(1, 'h')


def test_datetime64_to_timestamp(dt64):
    """https://www.epochconverter.com/
    """
    ts = utils.datetime64_to_timestamp(dt64)
    assert np.all(ts == 946684800 + np.arange(10) * 60 * 60)


def test_datetime64_to_datetime(dt64):
    dt = utils.datetime64_to_datetime(dt64)
    assert all([d.year == 2000 for d in dt])
    assert all([d.month == 1 for d in dt])
    assert all([d.day == 1 for d in dt])
    assert all([d.hour == i for i, d in enumerate(dt)])
    assert all([d.minute == 0 for d in dt])
    assert all([d.second == 0 for d in dt])
