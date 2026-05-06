import sys
sys.path.append("../../")

import pandas as pd
from ext_kinematic_casadi import Ekinematic
import f110
import numpy as np
from sklearn.preprocessing import StandardScaler


# Initialize the model with parameters
params = f110.F110()
# Initialize the kinematic model with the parameters
model = Ekinematic(params)

def prepare_dataset_from_raw_pure_pursuit(map_name, SAMPLING_TIME=0.02, mu=1.0489, diff_directory=False, cbf_mu=False):

    # Load the data from data/raw/ directory (relative to src/data_preprocess/)
    data_path = f'../../data/raw/traj_MPPI_{map_name}_mu_{mu}.csv'
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, comment='#')

    # Drop rows where C_alpha_smooth is NaN (typically the first few rows before window fills)
    df = df.dropna(subset=['C_alpha_smooth']).reset_index(drop=True)

            
    # Getting the next state values
    df['S_x_next'] = df['S_x'].shift(-1)
    df['S_y_next'] = df['S_y'].shift(-1)
    df['S_delta_next'] = df['S_delta'].shift(-1)
    df['S_vel_next'] = df['S_vel'].shift(-1)
    df['S_psi_next'] = df['S_psi'].shift(-1)
    df['S_omega_next'] = df['S_omega'].shift(-1)
    df['S_beta_next'] = df['S_beta'].shift(-1)
    # Removing the last row as it contains NaN values for the next state
    df = df[:-1]

    # We compute the next state values using ekinematic model. The difference between the next state values will be used as 'target' for GP mdoel.
    pred_xs, pred_ys, pred_deltas, pred_vels, pred_psis, pred_omegas, pred_betas = [], [], [], [], [], [], []
    for i in range(len(df) ):
        s = df.iloc[i][['S_x', 'S_y', 'S_delta', 'S_vel', 'S_psi', 'S_omega', 'S_beta']].values
        u = df.iloc[i][['A_a', 'A_sv']].values

        dxdt = model.casadi(s, u)

        pred_x = s[0] + (dxdt[0]*SAMPLING_TIME)
        pred_y = s[1] + (dxdt[1]*SAMPLING_TIME)
        pred_delta = s[2] + (dxdt[2]*SAMPLING_TIME)
        pred_vel = s[3] + (dxdt[3]*SAMPLING_TIME)
        pred_psi = s[4] + (dxdt[4]*SAMPLING_TIME)
        pred_omega = s[5] + (dxdt[5]*SAMPLING_TIME)
        pred_beta = s[6] + (dxdt[6]*SAMPLING_TIME)
        
        if pred_psi > 2 * np.pi:
            pred_psi -= 2 * np.pi
        elif pred_psi < 0:
            pred_psi += 2 * np.pi

        pred_xs.append(pred_x)
        pred_ys.append(pred_y)
        pred_deltas.append(pred_delta)
        pred_vels.append(pred_vel)
        pred_psis.append(pred_psi)
        pred_omegas.append(pred_omega)
        pred_betas.append(pred_beta)

    df['pred_x'] = pred_xs
    df['pred_y'] = pred_ys
    df['pred_delta'] = pred_deltas
    df['pred_vel'] = pred_vels
    df['pred_psi'] = pred_psis
    df['pred_omega'] = pred_omegas
    df['pred_beta'] = pred_betas


    # Get the MSE for each state variable - Confirming that the disconnect is present in omega and beta and not in the others.
    mse_x = ((df['S_x_next'] - df['pred_x']) ** 2).mean()
    mse_y = ((df['S_y_next'] - df['pred_y']) ** 2).mean()
    mse_delta = ((df['S_delta_next'] - df['pred_delta']) ** 2).mean()
    mse_vel = ((df['S_vel_next'] - df['pred_vel']) ** 2).mean()
    mse_psi = ((df['S_psi_next'] - df['pred_psi']) ** 2).mean()
    mse_omega = ((df['S_omega_next'] - df['pred_omega']) ** 2).mean()
    mse_beta = ((df['S_beta_next'] - df['pred_beta']) ** 2).mean()
    print(f"MSE for each state variable: {mse_x.round(10)}, {mse_y.round(10)}, {mse_delta}, {mse_vel.round(10)}, {mse_psi.round(10)}, {mse_omega}, {mse_beta}") # This confirms that the disconnect is present in omega and beta, not in the others.

    # Compute the difference in omega and beta values that is the y columns for the GP models
    df['Y_x'] = df['S_x_next'] - df['pred_x']
    df['Y_y'] = df['S_y_next'] - df['pred_y']
    df['Y_delta'] = df['S_delta_next'] - df['pred_delta']
    df['Y_vel'] = df['S_vel_next'] - df['pred_vel']
    df['Y_psi'] = df['S_psi_next'] - df['pred_psi']
    df['Y_omega'] = df['S_omega_next'] - df['pred_omega']
    df['Y_beta'] = df['S_beta_next'] - df['pred_beta']
    # Remove columns ['S_x_next', 'S_y_next', 'S_delta_next', 'S_vel_next', 'S_psi_next', 'pred_x', 'pred_y', 'pred_delta', 'pred_vel', 'pred_psi'] from df
    df.drop(columns=['S_x_next', 'S_y_next', 'S_delta_next', 'S_vel_next', 'S_psi_next', 'pred_x', 'pred_y', 'pred_delta', 'pred_vel', 'pred_psi'], inplace=True)
    # Remove the columns ['S_omega_next', 'S_beta_next', 'pred_omega', 'pred_beta'] from df
    df.drop(columns=['S_omega_next', 'S_beta_next', 'pred_omega', 'pred_beta'], inplace=True)

    return df


def prepare_dataset_from_raw_pure_pursuit_df_input(df, SAMPLING_TIME=0.02):
    df = df.copy()  # own a fresh frame so column assignments don't warn (SettingWithCopy)
    # Getting the next state values
    df['S_x_next'] = df['S_x'].shift(-1)
    df['S_y_next'] = df['S_y'].shift(-1)
    df['S_delta_next'] = df['S_delta'].shift(-1)
    df['S_vel_next'] = df['S_vel'].shift(-1)
    df['S_psi_next'] = df['S_psi'].shift(-1)
    df['S_omega_next'] = df['S_omega'].shift(-1)
    df['S_beta_next'] = df['S_beta'].shift(-1)
    # Removing the last row as it contains NaN values for the next state
    df = df.iloc[:-1].copy()

    # We compute the next state values using ekinematic model. The difference between the next state values will be used as 'target' for GP mdoel.
    pred_xs, pred_ys, pred_deltas, pred_vels, pred_psis, pred_omegas, pred_betas = [], [], [], [], [], [], []
    for i in range(len(df) ):
        s = df.iloc[i][['S_x', 'S_y', 'S_delta', 'S_vel', 'S_psi', 'S_omega', 'S_beta']].values
        u = df.iloc[i][['A_a', 'A_sv']].values

        dxdt = model.casadi(s, u)

        pred_x = s[0] + (dxdt[0]*SAMPLING_TIME)
        pred_y = s[1] + (dxdt[1]*SAMPLING_TIME)
        pred_delta = s[2] + (dxdt[2]*SAMPLING_TIME)
        pred_vel = s[3] + (dxdt[3]*SAMPLING_TIME)
        pred_psi = s[4] + (dxdt[4]*SAMPLING_TIME)
        pred_omega = s[5] + (dxdt[5]*SAMPLING_TIME)
        pred_beta = s[6] + (dxdt[6]*SAMPLING_TIME)
        
        if pred_psi > 2 * np.pi:
            pred_psi -= 2 * np.pi
        elif pred_psi < 0:
            pred_psi += 2 * np.pi

        pred_xs.append(pred_x)
        pred_ys.append(pred_y)
        pred_deltas.append(pred_delta)
        pred_vels.append(pred_vel)
        pred_psis.append(pred_psi)
        pred_omegas.append(pred_omega)
        pred_betas.append(pred_beta)

    df['pred_x'] = pred_xs
    df['pred_y'] = pred_ys
    df['pred_delta'] = pred_deltas
    df['pred_vel'] = pred_vels
    df['pred_psi'] = pred_psis
    df['pred_omega'] = pred_omegas
    df['pred_beta'] = pred_betas


    # Get the MSE for each state variable - Confirming that the disconnect is present in omega and beta and not in the others.
    mse_x = ((df['S_x_next'] - df['pred_x']) ** 2).mean()
    mse_y = ((df['S_y_next'] - df['pred_y']) ** 2).mean()
    mse_delta = ((df['S_delta_next'] - df['pred_delta']) ** 2).mean()
    mse_vel = ((df['S_vel_next'] - df['pred_vel']) ** 2).mean()
    mse_psi = ((df['S_psi_next'] - df['pred_psi']) ** 2).mean()
    mse_omega = ((df['S_omega_next'] - df['pred_omega']) ** 2).mean()
    mse_beta = ((df['S_beta_next'] - df['pred_beta']) ** 2).mean()
    print(f"MSE for each state variable: {mse_x.round(10)}, {mse_y.round(10)}, {mse_delta}, {mse_vel.round(10)}, {mse_psi.round(10)}, {mse_omega}, {mse_beta}") # This confirms that the disconnect is present in omega and beta, not in the others.

    # Compute the difference in omega and beta values that is the y columns for the GP models
    df['Y_x'] = df['S_x_next'] - df['pred_x']
    df['Y_y'] = df['S_y_next'] - df['pred_y']
    df['Y_delta'] = df['S_delta_next'] - df['pred_delta']
    df['Y_vel'] = df['S_vel_next'] - df['pred_vel']
    df['Y_psi'] = df['S_psi_next'] - df['pred_psi']
    df['Y_omega'] = df['S_omega_next'] - df['pred_omega']
    df['Y_beta'] = df['S_beta_next'] - df['pred_beta']
    # Remove columns ['S_x_next', 'S_y_next', 'S_delta_next', 'S_vel_next', 'S_psi_next', 'pred_x', 'pred_y', 'pred_delta', 'pred_vel', 'pred_psi'] from df
    df.drop(columns=['S_x_next', 'S_y_next', 'S_delta_next', 'S_vel_next', 'S_psi_next', 'pred_x', 'pred_y', 'pred_delta', 'pred_vel', 'pred_psi'], inplace=True)
    # Remove the columns ['S_omega_next', 'S_beta_next', 'pred_omega', 'pred_beta'] from df
    df.drop(columns=['S_omega_next', 'S_beta_next', 'pred_omega', 'pred_beta'], inplace=True)

    return df


Y_COLUMNS = ['Y_x', 'Y_y', 'Y_delta', 'Y_vel', 'Y_psi', 'Y_omega', 'Y_beta']


def load_data(map_name, varidx, SAMPLING_TIME=0.02, xscaler=None, yscaler=None, mu=1.0489):

    # Getting the processed dataset. This dataset contains the State, Action and difference between f_dyn and f_kin states i.e. y states
    df = prepare_dataset_from_raw_pure_pursuit(map_name, SAMPLING_TIME, mu)

    # Based on the variable index, we select the columns for x and y.
    # I have not included S_x, S_psi and S_y as they will cause issues in generalizing the model to differnet tracks.
    # The same was done in bayes race paper. Achin Jain et al. "Bayesian Optimization for Learning Driving Policies in Autonomous Racing", 2020.
    X = df[['S_delta', 'S_vel', 'S_omega', 'S_beta', 'A_a', 'A_sv', 'C_alpha_smooth']].values
    # Get the Y column by name based on the variable index
    y = df[Y_COLUMNS[varidx]].values

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(X)
        yscaler.fit(y.reshape(-1, 1))
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler
    else:
        print("Using the provided scalers.")
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler

def load_data_diff_directory(map_name, varidx, SAMPLING_TIME=0.02, xscaler=None, yscaler=None, mu=1.0489):

    # Getting the processed dataset. This dataset contains the State, Action and difference between f_dyn and f_kin states i.e. y states
    df = prepare_dataset_from_raw_pure_pursuit(map_name, SAMPLING_TIME, mu, diff_directory=True)

    X = df[['S_delta', 'S_vel', 'S_omega', 'S_beta', 'A_a', 'A_sv', 'C_alpha_smooth']].values
    y = df[Y_COLUMNS[varidx]].values

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(X)
        yscaler.fit(y.reshape(-1, 1))
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler
    else:
        print("Using the provided scalers.")
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler

def load_data_slow_data(map_name, varidx, SAMPLING_TIME=0.02, xscaler=None, yscaler=None, mu=1.0489):

    # Getting the processed dataset. This dataset contains the State, Action and difference between f_dyn and f_kin states i.e. y states
    df = prepare_dataset_from_raw_pure_pursuit(map_name, SAMPLING_TIME, mu, diff_directory=False, cbf_mu=True)

    X = df[['S_delta', 'S_vel', 'S_omega', 'S_beta', 'A_a', 'A_sv', 'C_alpha_smooth']].values
    y = df[Y_COLUMNS[varidx]].values

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(X)
        yscaler.fit(y.reshape(-1, 1))
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler
    else:
        print("Using the provided scalers.")
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler



def load_data_curved_road(map_name, varidx, SAMPLING_TIME=0.02, xscaler=None, yscaler=None, mu=1.0489):
    LH = 20 # Lookahead steps

    # Getting the processed dataset. This dataset contains the State, Action and difference between f_dyn and f_kin states i.e. y states
    df = prepare_dataset_from_raw_pure_pursuit(map_name, SAMPLING_TIME, mu)

    # Compute lookahead distance
    df['L'] = np.sqrt((df['S_x'].shift(-LH) - df['S_x'])**2 + (df['S_y'].shift(-LH) - df['S_y'])**2)
    df['lateral_distance'] = np.abs(df['S_y'].shift(-LH) - df['S_y'])
    df['curvature'] = (2 * df['lateral_distance']) / df['L']**2

    df = df[df['curvature'] >= 0.49]

    X = df[['S_delta', 'S_vel', 'S_omega', 'S_beta', 'A_a', 'A_sv', 'C_alpha_smooth']].values
    y = df[Y_COLUMNS[varidx]].values

    if xscaler is None or yscaler is None:
        xscaler = StandardScaler()
        yscaler = StandardScaler()
        xscaler.fit(X)
        yscaler.fit(y.reshape(-1, 1))
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler
    else:
        print("Using the provided scalers.")
        return xscaler.transform(X), yscaler.transform(y.reshape(-1, 1)), xscaler, yscaler
