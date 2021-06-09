if __name__ == "__main__":
    import numpy as np
    from ttools.create_dataset import process_dataset
    output_dir = "E:\\tec_data"
    start_year = "2010"
    end_year = "2020"
    START_YEAR = np.datetime64(start_year)
    END_YEAR = np.datetime64(end_year)

    # configure grid
    MLAT_BINS = np.arange(29.5, 90)
    MLT_BINS = np.arange(-12, 12 + 24 / 360, 48 / 360)
    process_dataset(START_YEAR, END_YEAR, MLAT_BINS, MLT_BINS, map_period=np.timedelta64(1, 'h'), output_dir=output_dir)
