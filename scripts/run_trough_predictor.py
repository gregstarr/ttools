import numpy as np
import os

from ttools import io, utils, config, trough_labeling

if __name__ == "__main__":
    RERUN = False
    output_fn = "E:\\dataset2.npz"

    if not RERUN and os.path.exists(output_fn):
        trough_data = np.load(output_fn)
        start_date = trough_data['time'][-1] + np.timedelta64(1, 'h')
        troughs = [trough_data['trough']]
        ssmlons = [trough_data['ssmlon']]
        times = [trough_data['time']]
        xs = [trough_data['x']]
    else:
        start_date = np.datetime64("2010-01-01T00:00:00")
        troughs = []
        ssmlons = []
        times = []
        xs = []

    end_date = np.datetime64("2020-01-01T00:00:00")
    one_day = np.timedelta64(1, 'D')
    dates = np.arange(start_date, end_date, one_day)
    print(f"START DATE: {start_date}")
    print(f"END DATE: {end_date}")

    job_manager = trough_labeling.TroughLabelJobManager(trough_labeling.RbfInversionLabelJob)

    for date in dates:
        job = job_manager.make_job(date)
        job.run()
        troughs.append(job.trough)
        ssmlons.append(job.ssmlon)
        times.append(job.times)
        xs.append(job.x)

        np.savez(output_fn,
                 x=np.concatenate(xs, axis=0),
                 trough=np.concatenate(troughs, axis=0),
                 ssmlon=np.concatenate(ssmlons, axis=0),
                 time=np.concatenate(times, axis=0))
