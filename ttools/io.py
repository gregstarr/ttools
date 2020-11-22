import numpy as np
import datetime
import pandas

OMNI_COLUMNS = [
    "rotation_number",
    "imf_id",
    "sw_id",
    "imf_n",
    "plasma_n",
    "b_mag",
    "b_vector_mag",
    "b_vector_lat_avg",
    "b_vector_lon_avg",
    "bx",
    "by_gse",
    "bz_gse",
    "by_gsm",
    "bz_gsm",
    "b_mag_std",
    "b_vector_mag_std",
    "bx_std",
    "by_std",
    "bz_std",
    "proton_temp",
    "proton_density",
    "plasma_speed",
    "plasma_lon_angle",
    "plasma_lat_angle",
    "na_np_ratio",
    "flow_pressure",
    "temp_std",
    "density_std",
    "speed_std",
    "phi_v_std",
    "theta_v_std",
    "na_np_ratio_std",
    "e_field",
    "plasma_beta",
    "alfven_mach_number",
    "kp",
    "r",
    "dst",
    "ae",
    "proton_flux_1",
    "proton_flux_2",
    "proton_flux_4",
    "proton_flux_10",
    "proton_flux_30",
    "proton_flux_60",
    "proton_flux_flag",
    "ap",
    "f107",
    "pcn",
    "al",
    "au",
    "magnetosonic_mach_number",
]


def get_gm_index_kyoto(fn="E:\\2000_2020_kp_ap.txt"):
    with open(fn, 'r') as f:
        text = f.readlines()
    ut_list = []
    kp_list = []
    ap_list = []
    for line in text[1:]:
        day = datetime.datetime.strptime(line[:8], '%Y%m%d')
        dt = datetime.timedelta(hours=3)
        uts = np.array([(day + i * dt).timestamp() for i in range(8)], dtype=int)
        kp = []
        for i in range(9, 25, 2):
            num = float(line[i])
            sign = line[i + 1]
            if sign == '+':
                num += 1 / 3
            elif sign == '-':
                num -= 1 / 3
            kp.append(num)
        kp_sum = float(line[25:27])
        sign = line[27]
        if sign == '+':
            kp_sum += 1 / 3
        elif sign == '-':
            kp_sum -= 1 / 3
        assert abs(kp_sum - sum(kp)) < .01
        kp_list.append(kp)
        ap = []
        for i in range(28, 52, 3):
            ap.append(float(line[i:i + 3]))
        ap = np.array(ap, dtype=int)
        Ap = float(line[52:55])
        ut_list.append(uts)
        ap_list.append(ap)

    ut = np.concatenate(ut_list)
    ap = np.concatenate(ap_list)
    kp = np.concatenate(kp_list)
    return pandas.DataFrame({'kp': kp, 'ap': ap, 'ut': ut}, index=pandas.to_datetime(ut, unit='s'))


def get_omni_data(fn="E:\\omni2_all_years.dat"):
    data = np.loadtxt(fn)
    year = (data[:, 0] - 1970).astype('datetime64[Y]')
    doy = (data[:, 1] - 1).astype('timedelta64[D]')
    hour = data[:, 2].astype('timedelta64[h]')
    datetimes = (year + doy + hour).astype('datetime64[s]')
    dtindex = pandas.DatetimeIndex(datetimes)
    df = pandas.DataFrame(data=data[:, 3:], index=dtindex, columns=OMNI_COLUMNS)
    for field in df:
        bad_val = df[field].max()
        bad_val_str = str(int(np.floor(bad_val)))
        if bad_val_str.count('9') == len(bad_val_str):
            mask = df[field] == bad_val
            df[field].loc[mask] = np.nan
    return df


def get_borovsky_data(fn="E:\\borovsky_2020_data.txt"):
    data = np.loadtxt(fn, skiprows=1)
    year = (data[:, 1] - 1970).astype('datetime64[Y]')
    doy = (data[:, 2] - 1).astype('timedelta64[D]')
    hour = data[:, 3].astype('timedelta64[h]')
    datetimes = (year + doy + hour).astype('datetime64[s]')
    return datetimes.astype(int), data[:, 4:]


if __name__ == "__main__":
    pass
