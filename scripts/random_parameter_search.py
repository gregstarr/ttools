if __name__ == "__main__":
    import time
    import os
    import pandas

    from ttools import compare, io

    N_EXPERIMENTS = 100  # number of random parameter settings
    N_TRAILS = 20  # number of trials for each parameter setting
    base_dir = "E:\\trough_comparison"

    processed_results = []
    for i in range(N_EXPERIMENTS):
        # setup directory
        experiment_dir = os.path.join(base_dir, f"experiment_{i}")
        os.makedirs(experiment_dir, exist_ok=True)

        # get and save hyperparameter list
        params = compare.get_random_hyperparams()
        print(params)
        io.write_yaml(os.path.join(experiment_dir, 'params.yaml'), **params)

        t0 = time.time()
        results = compare.run_n_random_days(N_TRAILS, **params)
        print(compare.process_results(results))
        results.to_csv(os.path.join(experiment_dir, "results.csv"))
        tf = time.time()
        print(f"THAT TOOK {(tf - t0) / 60} MINUTES")
        processed_results.append(compare.process_results(results))
        pandas.DataFrame(processed_results).to_csv(os.path.join(base_dir, "results.csv"))
