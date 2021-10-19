if __name__ == "__main__":
    import time
    import os
    from ttools import compare, utils, trough_labeling

    tlm = {
        'tlm1': trough_labeling.RbfInversionLabelJob,
        # 'tlm2': tl.ImageProcessingLabelJob,
        'baseline': trough_labeling.AaBaselineLabelJob,
        # 'constant_baseline': tl.ConstantBaselineLabelJob,
        'model_baseline': trough_labeling.ModelBaselineLabelJob,
    }
    N = 60
    comparison_manager = compare.ComparisonManager.random_dates(N)

    for name, class_ in tlm.items():
        t0 = time.time()
        os.makedirs(f"E:\\results\\method_{name}", exist_ok=True)
        job_manager = trough_labeling.TroughLabelJobManager(class_)
        results = comparison_manager.run_comparison(job_manager)
        results.to_csv(f"E:\\results\\method_{name}\\results.csv")
        tf = time.time()
        minutes = (tf - t0) / 60
        with open(f"E:\\results\\method_{name}\\time.txt", 'w') as f:
            f.write(f"{N}\n{minutes}")
        print(f"THAT TOOK {minutes} MINUTES")
