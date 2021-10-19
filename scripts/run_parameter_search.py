if __name__ == "__main__":
    from ttools import compare
    from ttools.trough_labeling import *

    N_TRAILS = 25  # number of trials for each parameter setting
    N_EXPERIMENTS = 200

    default_params = {
        'bg_est_shape': (1, 17, 17),
        'model_weight_max': 15,
        'rbf_bw': 1,
        'tv_hw': 2,
        'tv_vw': 1,
        'l2_weight': .05,
        'tv_weight': .15,
        'perimeter_th': 40,
        'area_th': 40,
        'artifact_key': '7',
        'auroral_boundary': True,
        'prior_order': 1,
        'prior': 'auroral_boundary',
        'prior_arb_offset': -3
    }

    grid_search_params = {
        'bg_est_shape': [(1, 15, 15), (1, 17, 17), (1, 19, 19)],
        'tv_hw': [2, 3],
        'l2_weight': [.02, .05, .07],
        'tv_weight': [.1, .15, .2],
        'artifact_key': ['3', '5', '7'],
    }

    # compare.grid_parameter_search(default_params, grid_search_params, N_TRAILS)
    compare.random_parameter_search(RbfInversionLabelJob, N_EXPERIMENTS, N_TRAILS)
    # compare.random_parameter_search(ImageProcessingLabelJob, N_EXPERIMENTS, N_TRAILS)
