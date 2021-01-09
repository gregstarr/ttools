import apexpy
import numpy as np


def geo_to_mlt_grid(lat, lon, times, converter=None, height=0, ssheight=50*6371):
    """Convert a grid of geographic coordinates at different times to MLAT / MLT coordinates.

    Parameters
    ----------
    lat: numpy.ndarray
    lon: numpy.ndarray
    times: numpy.ndarray
            times as timestamp or datetime64
    converter: apexpy.Apex (optional)
    height: numeric
    ssheight: numeric

    Returns
    -------
    mlat, mlt: numpy.ndarray[float]
    """
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mlat, mlon = converter.geo2apex(lat_grid.ravel(), lon_grid.ravel(), height)
    mlat = mlat.reshape(lat_grid.shape)
    mlon = mlon.reshape(lat_grid.shape)
    mlat, mlt, ssmlon = mlon_to_mlt_grid(mlat, mlon, times, converter, ssheight)
    return mlat, mlt


def mlon_to_mlt_grid(mlat_grid, mlon_grid, times, converter=None, ssheight=50*6371):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    ssglat, ssglon = subsol_array(times)
    ssmlat, ssmlon = converter.geo2apex(ssglat, ssglon, ssheight)
    mlt = (180 + mlon_grid[None, :, :] - ssmlon[:, None, None]) / 15 % 24
    mlat = mlat_grid[None, :, :] * np.ones((times.shape[0], 1, 1), dtype=float)
    return mlat, mlt, ssmlon


def geo_to_mlt(glat, glon, height, times, converter=None, ssheight=50*6371):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    mlat, mlon = converter.geo2apex(glat, glon, height)
    ssglat, ssglon = subsol_array(times)
    ssmlat, ssmlon = converter.geo2apex(ssglat, ssglon, ssheight)
    mlt = (180 + mlon - ssmlon) / 15 % 24
    return mlat, mlt


def mlon_to_mlt_array(mlon, times, converter=None, ssheight=50*6371):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    ssglat, ssglon = subsol_array(times)
    ssmlat, ssmlon = converter.geo2apex(ssglat, ssglon, ssheight)
    mlt = (180 + mlon - ssmlon) / 15 % 24
    return mlt


def subsol_array(times):
    """get the subsolar point in geocentric coordinates for an array of times

    Parameters
    ----------
    times: numpy.ndarray[datetime64]

    Returns
    -------
    sslat, sslon: numpy.ndarray[float]
    """
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    # convert to year, day of year and seconds since midnight
    year_floor = times.astype('datetime64[Y]')
    day_floor = times.astype('datetime64[D]')
    year = year_floor.astype(int) + 1970
    doy = (day_floor - year_floor).astype(int) + 1
    sec = (times.astype('datetime64[s]') - day_floor).astype(float)

    if not np.all(1601 <= year) and np.all(year <= 2100):
        raise ValueError('Year must be in [1601, 2100]')

    yr = year - 2000

    nleap = np.floor((year - 1601.0) / 4.0).astype(int)
    nleap -= 99
    mask_1900 = year <= 1900
    if np.any(mask_1900):
        ncent = np.floor((year[mask_1900] - 1601.0) / 100.0).astype(int)
        ncent = 3 - ncent[mask_1900]
        nleap[mask_1900] = nleap[mask_1900] + ncent

    l0 = -79.549 + (-0.238699 * (yr - 4.0 * nleap) + 3.08514e-2 * nleap)
    g0 = -2.472 + (-0.2558905 * (yr - 4.0 * nleap) - 3.79617e-2 * nleap)
    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (sec / 86400.0 - 1.5) + doy
    # Mean longitude of Sun:
    lmean = l0 + 0.9856474 * df
    # Mean anomaly in radians:
    grad = np.radians(g0 + 0.9856003 * df)
    # Ecliptic longitude:
    lmrad = np.radians(lmean + 1.915 * np.sin(grad) + 0.020 * np.sin(2.0 * grad))
    sinlm = np.sin(lmrad)
    # Obliquity of ecliptic in radians:
    epsrad = np.radians(23.439 - 4e-7 * (df + 365 * yr + nleap))
    # Right ascension:
    alpha = np.degrees(np.arctan2(np.cos(epsrad) * sinlm, np.cos(lmrad)))
    # Declination, which is also the subsolar latitude:
    sslat = np.degrees(np.arcsin(np.sin(epsrad) * sinlm))
    # Equation of time (degrees):
    etdeg = lmean - alpha
    nrot = np.round(etdeg / 360.0)
    etdeg = etdeg - 360.0 * nrot
    # Subsolar longitude:
    sslon = 180.0 - (sec / 240.0 + etdeg)  # Earth rotates one degree every 240 s.
    nrot = np.round(sslon / 360.0)
    sslon = sslon - 360.0 * nrot
    return sslat, sslon
