import numpy as np
import datetime


def datetime64_to_datetime(dt64):
    """convert a numpy.datetime64 to a datetime.datetime

    Parameters
    ----------
    dt64: numpy.datetime64
            the datetime64 to convert

    Returns
    -------
    datetime.datetime
            the converted object
    """
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)


def timestamp_to_datetime(ts):
    return datetime.datetime.utcfromtimestamp(ts)
