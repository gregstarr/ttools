from scipy import stats


PARAM_SAMPLING = {
    'tv_weight': stats.loguniform(.001, 1),
    'l2_weight': stats.loguniform(.001, 1),
    'bge_spatial_size': stats.randint(3, 13),
    'bge_temporal_rad': stats.randint(0, 3),
    'rbf_bw': stats.randint(1, 4),
    'tv_hw': stats.randint(1, 4),
    'tv_vw': stats.randint(1, 4),
    'model_weight_max': stats.randint(4, 30),
    'perimeter_th': stats.randint(10, 100),
    'area_th': stats.randint(10, 100),
}


def get_random_hyperparams():
    params = {}
    bge_temporal_size = 0
    bge_spatial_size = 0
    for p in PARAM_SAMPLING:
        try:
            val = PARAM_SAMPLING[p].rvs().item()
        except:
            val = PARAM_SAMPLING[p].rvs()
        if p == 'bge_temporal_rad':
            bge_temporal_size = val * 2 + 1
        elif p == 'bge_spatial_size':
            bge_spatial_size = val * 2 + 1
        else:
            params[p] = val
    params['bg_est_shape'] = (bge_temporal_size, bge_spatial_size, bge_spatial_size)
    return params


if __name__ == "__main__":
    import time
    import os
    import pandas

    from ttools import compare, io

    N_EXPERIMENTS = 20
    N_TRAILS = 1
    base_dir = "E:\\trough_comparison"

    processed_results = []
    for i in range(N_EXPERIMENTS):
        # setup directory
        experiment_dir = os.path.join(base_dir, f"experiment_{i}")
        os.makedirs(experiment_dir, exist_ok=True)

        # get and save hyperparameter list
        params = get_random_hyperparams()
        print(params)
        io.write_yaml(os.path.join(experiment_dir, 'params.yaml'), **params)

        t0 = time.time()
        results = compare.run_n_random_days(N_TRAILS, **params)
        print(compare.process_results(results))
        results.to_csv(os.path.join(experiment_dir, "results.csv"))
        tf = time.time()
        print(f"THAT TOOK {(tf - t0) / 60} MINUTES")
        processed_results.append(compare.process_results(results))
    processed_results = pandas.DataFrame(processed_results)
    processed_results.to_csv(os.path.join(base_dir, "results.csv"))
