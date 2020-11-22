import numpy as np
import datetime


def datetime64_to_timestamp(dt64):
    return (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def datetime64_to_datetime(dt64):
    ts = datetime64_to_timestamp(dt64)
    return timestamp_to_datetime(ts)


def timestamp_to_datetime(ts):
    return datetime.datetime.utcfromtimestamp(ts)


def get_random_map_id(start_time=np.datetime64("2013-12-03T00:00:00"), end_time=np.datetime64("2019-12-30T00:00:00")):
    dset_range = (end_time.astype('datetime64[h]') - start_time.astype('datetime64[h]')).astype(int)
    hours_offset = np.random.randint(0, dset_range)
    map_time = start_time + np.timedelta64(hours_offset, 'h')
    year = map_time.astype('datetime64[Y]').astype(int) + 1970
    month = map_time.astype('datetime64[M]').astype(int) % 12 + 1
    index = (map_time.astype('datetime64[h]') - map_time.astype('datetime64[M]')).astype(int)
    return year, month, index