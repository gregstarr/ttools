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
        'bg_est_shape': (3, 15, 15),
        'model_weight_max': 20,
        'rbf_bw': 1,
        'tv_hw': 1,
        'tv_vw': 1,
        'l2_weight': .08,
        'tv_weight': .05,
        'perimeter_th': 50,
        'area_th': 20,
    }
    t0 = time.time()
    results = compare.run_n_random_days(10, **params)
    print(compare.process_results(results, bad_mlon_range=[70, 125]))
    results.to_csv("E:\\temp_comparison_results\\results.csv")
    tf = time.time()
    print(f"THAT TOOK {(tf - t0)/60} MINUTES")
