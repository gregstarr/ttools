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
        'bg_est_shape': (3, 21, 21),
        'model_weight_max': 5,
        'rbf_bw': 1,
        'tv_hw': 2,
        'tv_vw': .5,
        'l2_weight': .13,
        'tv_weight': .1,
        'perimeter_th': 20,
        'area_th': 40,
        'artifact_key': '9',
        'auroral_boundary': True,
        'prior_order': 1,
        'prior': 'empirical',
        'prior_arb_offset': -1
    }
    t0 = time.time()
    results = compare.run_n_random_days(100, **params, make_plots=True, plot_dir="E:\\temp_plots")
    diffs = compare.get_diffs(results, bad_mlon_range=[130, 260])
    stats = compare.process_results(results, bad_mlon_range=[130, 260])
    for k, v in stats.items():
        print(k, v)
    results.to_csv("E:\\temp_comparison_results\\results.csv")
    tf = time.time()
    print(f"THAT TOOK {(tf - t0)/60} MINUTES")
