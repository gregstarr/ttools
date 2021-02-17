"""
- configure parameters
- pick N random days
- run swarm and tec trough detection
- compile results
"""
if __name__ == "__main__":
    import time
    from ttools import compare

    params = {
        'bg_est_shape': (3, 17, 17),
        'model_weight_max': 30,
        'rbf_bw': 1,
        'tv_hw': 1,
        'tv_vw': 1,
        'l2_weight': .1,
        'tv_weight': .1,
        'perimeter_th': 50,
        'area_th': 50,
    }
    t0 = time.time()
    results = compare.run_n_random_days(100, **params, make_plots=True, plot_dir="E:\\temp_plots")
    stats = compare.process_results(results, bad_mlon_range=[65, 130])
    for k, v in stats.items():
        print(k, v)
    results.to_csv("E:\\temp_comparison_results\\results.csv")
    tf = time.time()
    print(f"THAT TOOK {(tf - t0)/60} MINUTES")
