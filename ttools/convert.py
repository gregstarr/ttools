import apexpy
import numpy as np


def geo_to_mlt_array(glat, glon, height, times, converter=None, ssheight=50*6371):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    mlat, mlon = converter.geo2apex(glat, glon, height)
    mlt = mlon_to_mlt_array(mlon, times, converter)
    return mlat, mlt


def mlon_to_mlt_array(mlon, times, converter=None, ssheight=50*6371, return_ssmlon=False):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    ssglat, ssglon = subsol_array(times)
    ssmlat, ssmlon = converter.geo2apex(ssglat, ssglon, ssheight)
    mlt = (180 + mlon - ssmlon) / 15 % 24
    if return_ssmlon:
        return mlt, ssmlon
    return mlt


def mlt_to_geo_array(mlat, mlt, times, height=0, converter=None, ssheight=50*6371):
    if times.dtype is not np.dtype('datetime64[s]'):
        times = times.astype('datetime64[s]')
    if converter is None:
        converter = apexpy.Apex()
    ssglat, ssglon = subsol_array(times)
    ssalat, ssalon = converter.geo2apex(ssglat, ssglon, ssheight)
    mlon = (15 * mlt - 180 + ssalon + 360) % 360
    glat, glon, _ = converter.apex2geo(mlat, mlon, height)
    return glat, glon


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
    ut = (times.astype('datetime64[s]') - day_floor).astype(float)

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
    df = (ut / 86400.0 - 1.5) + doy
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
    sslon = 180.0 - (ut / 240.0 + etdeg)  # Earth rotates one degree every 240 s.
    nrot = np.round(sslon / 360.0)
    sslon = sslon - 360.0 * nrot
    return sslat, sslon
