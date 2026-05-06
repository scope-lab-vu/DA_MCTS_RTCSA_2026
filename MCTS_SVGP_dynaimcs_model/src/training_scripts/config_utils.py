from pyDOE2 import lhs
import numpy as np
SEED = 42
np.random.seed(SEED)

# Below two functions are taken from PAML repo for obtaining trainiing task configurations
def latin_hypercube_sampling_maxi_min(n_tasks: int, dim: int, observed_configuration_space_interval):
    task_configurations = lhs(dim, n_tasks, criterion="centermaximin", random_state=SEED)
    lower_bound = observed_configuration_space_interval[:, 0] # This is the min of the task variable
    upper_bound = observed_configuration_space_interval[:, 1] # This is the min of the task variable (i.e. friction for us)
    task_configurations = (upper_bound - lower_bound) * task_configurations + lower_bound
    return task_configurations

def sample_from_multidim_interval_uniformly(bounds, size):
    # bounds is a 2D array where each row is [min, max] for that dimension
    samples = []
    for _ in range(size):
        parameters = []
        for bound in bounds:
            parameters.append(np.random.uniform(bound[0], bound[1]))
        samples.append(parameters)
    return np.array(samples).reshape(size, bounds.shape[0])


# Below is also taken from PAML repo for creating config grid for test tasks configurations
def create_configurations_grid(configuration_space_interval, grid_resolution: int = 4, config_space_decimals: int = 2):
    """
        configuration_space_interval: List of tuples [(min1, max1), (min2, max2), ...] for each dimension in the tasks configuration space. For just this will be (1,2) shape array for friction with min and max values
        grid_resolution: Number of points to sample in each dimension. Basically the number of test tasks we want to create.
        config_space_decimals: Number of decimal places to round the configurations to. 
    """
    grids = []
    for dim in configuration_space_interval:
        grid = np.linspace(dim[0], dim[1], grid_resolution)
        grid = np.round(grid, config_space_decimals)
        grids.append(grid)
    mesh = np.meshgrid(*grids)
    configurations = []
    for dim in mesh:
        transformed_dim = np.transpose(dim).flatten()
        configurations.append(transformed_dim)
    configurations = np.transpose(configurations)
    return configurations.reshape(-1, len(configuration_space_interval))