if __name__ == "__main__":
    import time
    from ttools import compare, utils
    from ttools.trough_labeling import RbfInversionLabelJob, ImageProcessingLabelJob, TroughLabelJobManager

    t0 = time.time()

    comparison_manager = compare.ComparisonManager.random_dates(100)#, plot_dir="E:\\temp_plots")
    job_manager = TroughLabelJobManager(RbfInversionLabelJob, threshold=None)
    results = comparison_manager.run_comparison(job_manager)
    stats = compare.process_results(results, bad_mlon_range=[130, 260])
    for k, v in stats.items():
        print(k, v)
    results.to_csv("E:\\temp_comparison_results\\results.csv")
    tf = time.time()
    print(f"THAT TOOK {(tf - t0)/60} MINUTES")
