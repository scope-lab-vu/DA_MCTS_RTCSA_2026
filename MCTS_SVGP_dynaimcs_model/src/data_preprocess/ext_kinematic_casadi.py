import math
import casadi as ca


class SteerSettings(object):
    def __init__(self, params):
        self.min = params['min_steer']
        self.max = params['max_steer']
        self.v_min = -params['max_steer_vel']
        self.v_max = params['max_steer_vel']


class AccSettings(object):
    def __init__(self, params):
        self.v_min = params['min_v']
        self.v_max = params['max_v']
        self.v_switch = params['switch_v']
        self.a_min = params['min_acc']
        self.a_max = params['max_acc']

class Ekinematic():
    def __init__(self, vehicle_params):
        self.params = vehicle_params
        self.g     = 9.81
        self.lf    = vehicle_params['lf']
        self.lr    = vehicle_params['lr']
        self.mass  = vehicle_params['mass']
        self.Iz    = vehicle_params['Iz']
        self.Csf   = vehicle_params['Csf']
        self.Csr   = vehicle_params['Csr']
        self.hcog  = vehicle_params['hcog']
        self.mu    = vehicle_params['mu']
        self.min_v = vehicle_params['min_v']
        self.max_v = vehicle_params['max_v']
        self.s_v   = vehicle_params['switch_v']

        self.max_acc     = vehicle_params['max_acc']
        self.min_acc     = vehicle_params['min_acc']
        self.max_steer   = vehicle_params['max_steer']
        self.min_steer   = vehicle_params['min_steer']
        self.max_steer_v = vehicle_params['max_steer_vel']
        self.min_steer_v = -vehicle_params['max_steer_vel']

        self.max_inputs  = vehicle_params['max_inputs']
        self.min_inputs  = vehicle_params['min_inputs']
        self.max_rates   = vehicle_params['max_rates']
        self.min_rates   = vehicle_params['min_rates']
        self.n_states    = 7
        self.n_inputs    = 2
        self.timestep     = 0.02

    def casadi(self, s, u, dxdt=None):
        x = s[0]  # x position
        y = s[1]  # y position
        delta = s[2]  # steering angle
        v = s[3]  # velocity
        psi = s[4]  # yaw angle
        omega = s[5]  # yaw rate
        beta = s[6]  # slip angle
        accl = u[0]  # steering velocity
        sv = u[1]  # acceleration


        x_dot = v * ca.cos(beta + psi)
        y_dot = v * ca.sin(beta + psi)
        delta_dot = sv
        v_dot = accl
        phi_dot = omega
        omega_dot = ((delta_dot * v) + (delta * v_dot)) / (self.lf + self.lr)
        beta_dot = (self.lf * ((delta_dot * v) + (delta * v_dot))) / (self.lf + self.lr)

        if dxdt is None:
            dxdt = [0] * self.n_states

        dxdt[0] = x_dot
        dxdt[1] = y_dot
        dxdt[2] = delta_dot
        dxdt[3] = v_dot
        dxdt[4] = phi_dot
        dxdt[5] = omega_dot
        dxdt[6] = beta_dot
        
        return dxdt