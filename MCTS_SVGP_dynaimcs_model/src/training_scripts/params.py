import numpy as np

def get_params(map_name='YasMarina', Ts=0.02, model_path='None'):
    # PARAMETERS
    parser= {}
    parser['seed'] = 42
    parser['n_initial_training_envs'] = 3
    parser['test_task_count'] = 1 # The plus one is there because bounds[0][1] is always included in the test tasks
    parser['observed_config_space_dim'] = 1
    parser['config_space_decimals'] = 4

    parser['n_active_tasks'] = parser['n_initial_training_envs']
    parser['initial_training_configurations'] = 'LHS'
    # parser['training_trajectory_length'] = 2000 # Probably needs to be tuned based on the env. We will also need to show the effect of this in ablation study.

    parser['env_name'] = 'f110'
    parser['map_name'] = map_name
    parser['dt'] = Ts

    # parser['latent_variable_inference_steps'] = 1000
    parser['bounds'] = np.array([[0.31, 1.0489]])
    parser['test_bounds'] = np.array([[0.38, 1.0489]])

    parser['v5'] = {
        'kname': 'rbf',
        'M': 55,
        'sigma_y': 0.012563,
        'dim_h': 2,
    }
    # hyperparameters for varidx 6
    parser['v6'] = {
        'kname': 'rbf',
        'M': 65,
        'sigma_y': 0.004679,
        'dim_h': 2,
    }

    # Below are best params for task count 3 and 1 test task
    # # hyperparameters for varidx 5
    # parser['v5'] = {
    #     'kname': 'rbf',
    #     'M': 65,
    #     'sigma_y': 0.005781,
    #     'dim_h': 3,
    # }
    # # hyperparameters for varidx 6
    # parser['v6'] = {
    #     'kname': 'rbf',
    #     'M': 80,
    #     'sigma_y': 0.002898,
    #     'dim_h': 3,
    # }

    # Below are best params for S = 5. 3 train and 1 test
    # hyperparameters for varidx 5
    # parser['v5'] = {
    #     'kname': 'rbf',
    #     'M': 55,
    #     'sigma_y': 0.01256277350380703,
    #     'dim_h': 2,
    # }

    # # if model_path contains substring f_10489_10159_06305
    # if model_path is not None and 'f_10489_10159_06305' in model_path:
    #     # hyperparameters for varidx 5
    #     parser['v5'] = {
    #         'kname': 'rbf',
    #         'M': 80,
    #         'sigma_y': 0.005100,
    #         'dim_h': 3,
    #     }
    #     parser['v6'] = {
    #         'kname': 'rbf',
    #         'M': 75,
    #         'sigma_y': 0.001337,
    #         'dim_h': 2,
    #     }
    # elif model_path is not None and 'f_10489_08817_05472' in model_path:
    #     # hyperparameters for varidx 5
    #     parser['v5'] = {
    #         'kname': 'rbf',
    #         'M': 70,
    #         'sigma_y': 0.009289,
    #         'dim_h': 3,
    #     }
    #     parser['v6'] = {
    #         'kname': 'rbf',
    #         'M': 65,
    #         'sigma_y': 0.004133,
    #         'dim_h': 3,
    #     }
    # elif model_path is not None and 'f_10489_09929_0881' in model_path:
    #     print("Using model path with f_10489_09929_0881")
    #     # hyperparameters for varidx 5
    #     parser['v5'] = {
    #         'kname': 'rbf',
    #         'M': 70,
    #         'sigma_y': 0.009289,
    #         'dim_h': 3,
    #     }
    #     parser['v6'] = {
    #         'kname': 'rbf',
    #         'M': 75,
    #         'sigma_y': 0.001685,
    #         'dim_h': 2,
    #     }
    # else:
    #     parser['v5'] = {
    #         'kname': 'rbf',
    #         'M': 70,
    #         'sigma_y': 0.009289,
    #         'dim_h': 3,
    #     }
    #     # hyperparameters for varidx 6
    #     parser['v6'] = {
    #         'kname': 'rbf',
    #         'M': 75,
    #         'sigma_y': 0.001981,
    #         'dim_h': 2,
    #     }

    return parser

