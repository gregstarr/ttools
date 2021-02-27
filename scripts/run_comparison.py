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
        'bg_est_shape': (1, 15, 15),
        'model_weight_max': 4,
        'rbf_bw': 2,
        'tv_hw': 2,
        'tv_vw': 1,
        'l2_weight': .3,
        'tv_weight': .05,
        'perimeter_th': 70,
        'area_th': 25,
        'artifact_key': '3',
        'auroral_boundary': True,
        'prior_order': 2,
        'prior': 'empirical_model',
        # 'prior_arb_offset': -1
    }
    t0 = time.time()
    results = compare.run_n_random_days(100, **params, make_plots=True, plot_dir="E:\\temp_plots")
    stats = compare.process_results(results, bad_mlon_range=[130, 260])
    for k, v in stats.items():
        print(k, v)
    results.to_csv("E:\\temp_comparison_results\\results.csv")
    tf = time.time()
    print(f"THAT TOOK {(tf - t0)/60} MINUTES")
