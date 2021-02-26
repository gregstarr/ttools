if __name__ == "__main__":
    from ttools import compare

    N_EXPERIMENTS = 100  # number of random parameter settings
    N_TRAILS = 20  # number of trials for each parameter setting
    compare.random_parameter_search(N_EXPERIMENTS, N_TRAILS)
