import numpy as np
import datetime


def get_gm_index(fn="E:\\2000_2020_kp_ap.txt"):
    with open(fn, 'r') as f:
        text = f.readlines()
    time_list = []
    ap_list = []
    for line in text[1:]:
        day = datetime.datetime.strptime(line[:8], '%Y%m%d')
        dt = datetime.timedelta(hours=3)
        times = np.array([(day + i * dt).timestamp() for i in range(8)], dtype=int)
        kp = []
        for i in range(9, 25, 2):
            kp.append(line[i:i+2])
        kp_sum = line[25:28]
        ap = []
        for i in range(28, 52, 3):
            ap.append(int(line[i:i+3]))
        ap = np.array(ap, dtype=int)
        Ap = int(line[52:55])
        time_list.append(times)
        ap_list.append(ap)
    return np.concatenate(time_list), np.concatenate(ap_list)