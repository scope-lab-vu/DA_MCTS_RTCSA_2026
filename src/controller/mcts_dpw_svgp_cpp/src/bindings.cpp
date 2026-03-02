#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mcts_native.hpp"
#include "f1tenth_simulator.hpp"
#include <memory>

namespace py = pybind11;

// Numpy array -> std::vector<double>
inline std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

// std::vector<double> -> numpy array
inline py::array_t<double> to_numpy(const std::vector<double>& vec) {
    return py::array_t<double>(static_cast<py::ssize_t>(vec.size()), vec.data());
}

// Numpy float32 array -> std::vector<float>
inline std::vector<float> to_float_vec(py::array_t<float> arr) {
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.size);
}

// Wrapper class that owns both the simulator and the DPW tree.
// Exposed to Python as a single object with a clean API.
class F1TenthDPWSvgp {
    std::unique_ptr<F1TenthSimulator> sim_;
    std::unique_ptr<DPWNative> model_;

  public:
    F1TenthDPWSvgp(py::array_t<double> initial_obs,
               double alpha, double beta, double K,
               // Vehicle params
               double mu, double C_Sf, double C_Sr,
               double lf, double lr, double h_cg,
               double mass, double I_z,
               double s_min, double s_max,
               double sv_min, double sv_max,
               double v_switch, double a_max,
               double v_min, double v_max,
               double veh_width, double veh_length,
               // Occupancy map (imaginary boundary)
               py::array_t<float> occ_grid,
               double occ_x_min, double occ_y_min,
               double occ_cell_size, int occ_nx, int occ_ny,
               // Occupancy map (real track boundary)
               py::array_t<float> occ_grid_real,
               // Cost params
               double speed_weight, double alpha_threshold,
               double v_ref, int rollout_horizon,
               double discount, double dt,
               bool normalize_reward,
               // SVGP params
               py::array_t<double> omega_data,
               py::array_t<double> beta_data,
               int svgp_M, int svgp_D_input, int svgp_D_hidden,
               double base_timestep,
               unsigned seed = 0) {

        auto obs = to_vec(initial_obs);

        VehicleParamsCpp vp;
        vp.mu       = mu;
        vp.C_Sf     = C_Sf;
        vp.C_Sr     = C_Sr;
        vp.lf       = lf;
        vp.lr       = lr;
        vp.h        = h_cg;
        vp.m        = mass;
        vp.I        = I_z;
        vp.s_min    = s_min;
        vp.s_max    = s_max;
        vp.sv_min   = sv_min;
        vp.sv_max   = sv_max;
        vp.v_switch = v_switch;
        vp.a_max    = a_max;
        vp.v_min    = v_min;
        vp.v_max    = v_max;
        vp.width    = veh_width;
        vp.length   = veh_length;

        OccupancyMapData occ;
        occ.grid      = to_float_vec(occ_grid);
        occ.x_min     = occ_x_min;
        occ.y_min     = occ_y_min;
        occ.cell_size = occ_cell_size;
        occ.nx        = occ_nx;
        occ.ny        = occ_ny;

        // Real boundary shares same grid metadata
        OccupancyMapData occ_real;
        occ_real.grid      = to_float_vec(occ_grid_real);
        occ_real.x_min     = occ_x_min;
        occ_real.y_min     = occ_y_min;
        occ_real.cell_size = occ_cell_size;
        occ_real.nx        = occ_nx;
        occ_real.ny        = occ_ny;

        CostParams cost;
        cost.speed_weight      = speed_weight;
        cost.alpha_threshold   = alpha_threshold;
        cost.v_ref             = v_ref;
        cost.rollout_horizon   = rollout_horizon;
        cost.discount_factor   = discount;
        cost.dt                = dt;
        cost.normalize_reward  = normalize_reward;

        // Build SVGP params from flat arrays packed in Python
        auto omega_buf = omega_data.request();
        auto beta_buf  = beta_data.request();
        auto omega_svgp = std::make_shared<SVGPParams>(
            SVGPParams::from_flat(static_cast<double*>(omega_buf.ptr),
                                  svgp_M, svgp_D_input, svgp_D_hidden));
        auto beta_svgp = std::make_shared<SVGPParams>(
            SVGPParams::from_flat(static_cast<double*>(beta_buf.ptr),
                                  svgp_M, svgp_D_input, svgp_D_hidden));

        sim_ = std::make_unique<F1TenthSimulator>(
            vp, occ, occ_real, cost,
            omega_svgp, beta_svgp, base_timestep, seed);
        sim_->set_state(obs);
        model_ = std::make_unique<DPWNative>(sim_.get(), obs, alpha, beta, K, seed);
    }

    void set_v_ref_window(py::array_t<double> window) {
        sim_->set_v_ref_window(to_vec(window));
    }

    void set_action_prior(py::array_t<double> mean, py::array_t<double> std) {
        sim_->set_action_prior(to_vec(mean), to_vec(std));
    }

    void set_rollout_horizon(int h) {
        sim_->set_rollout_horizon(h);
    }

    void set_centerline(py::array_t<double> cx, py::array_t<double> cy,
                        py::array_t<double> heading) {
        sim_->set_centerline(to_vec(cx), to_vec(cy), to_vec(heading));
    }

    void learn(int Nsim) { model_->learn(Nsim); }

    int learn_timed(double budget_s) { return model_->learn_timed(budget_s); }

    py::array_t<double> best_action() {
        return to_numpy(model_->best_action_vec());
    }

    void forward(py::array_t<double> action, py::array_t<double> new_state) {
        model_->forward(to_vec(action), to_vec(new_state));
        // Reset simulator to the new root state
        sim_->set_state(to_vec(new_state));
    }

    void set_state(py::array_t<double> state) {
        sim_->set_state(to_vec(state));
    }

    // Return root actions as list of dicts: [{action, visits, avg_reward}, ...]
    py::list get_root_actions() {
        auto infos = model_->get_root_actions();
        py::list result;
        for (auto& info : infos) {
            py::dict d;
            d["action"] = to_numpy(info.action);
            d["visits"] = info.visits;
            d["avg_reward"] = info.avg_reward;
            result.append(d);
        }
        return result;
    }

    // Walk tree following most-visited actions to get the best trajectory.
    // Returns list of dicts: [{state, action, avg_reward, visits}, ...]
    py::list get_best_trajectory(int max_depth = 50) {
        auto steps = model_->get_best_trajectory(max_depth);
        py::list result;
        for (auto& step : steps) {
            py::dict d;
            d["state"] = to_numpy(step.state);
            d["action"] = to_numpy(step.action);
            d["avg_reward"] = step.avg_reward;
            d["visits"] = step.visits;
            result.append(d);
        }
        return result;
    }
};

PYBIND11_MODULE(mcts_f1tenth_svgp_cpp, m) {
    m.doc() = "C++ MCTS DPW + SVGP dynamics for F1TENTH racing";

    py::class_<F1TenthDPWSvgp>(m, "F1TenthDPWSvgp")
        .def(py::init<py::array_t<double>,
                       double, double, double,
                       // Vehicle params (18)
                       double, double, double,
                       double, double, double,
                       double, double,
                       double, double,
                       double, double,
                       double, double,
                       double, double,
                       double, double,
                       // Occupancy map (imaginary)
                       py::array_t<float>,
                       double, double, double, int, int,
                       // Occupancy map (real)
                       py::array_t<float>,
                       // Cost params
                       double, double, double, int, double, double,
                       bool,
                       // SVGP params
                       py::array_t<double>, py::array_t<double>,
                       int, int, int, double,
                       unsigned>(),
             py::arg("initial_obs"),
             py::arg("alpha"), py::arg("beta"), py::arg("K"),
             // Vehicle params
             py::arg("mu"), py::arg("C_Sf"), py::arg("C_Sr"),
             py::arg("lf"), py::arg("lr"), py::arg("h"),
             py::arg("m"), py::arg("I"),
             py::arg("s_min"), py::arg("s_max"),
             py::arg("sv_min"), py::arg("sv_max"),
             py::arg("v_switch"), py::arg("a_max"),
             py::arg("v_min"), py::arg("v_max"),
             py::arg("width"), py::arg("length"),
             // Occupancy map (imaginary)
             py::arg("occ_grid"),
             py::arg("occ_x_min"), py::arg("occ_y_min"),
             py::arg("occ_cell_size"), py::arg("occ_nx"), py::arg("occ_ny"),
             // Occupancy map (real)
             py::arg("occ_grid_real"),
             // Cost params
             py::arg("speed_weight"), py::arg("alpha_threshold"),
             py::arg("v_ref"), py::arg("rollout_horizon"),
             py::arg("discount"), py::arg("dt"),
             py::arg("normalize_reward") = false,
             // SVGP params
             py::arg("omega_data"), py::arg("beta_data"),
             py::arg("svgp_M"), py::arg("svgp_D_input"), py::arg("svgp_D_hidden"),
             py::arg("base_timestep"),
             py::arg("seed") = 0)
        .def("set_v_ref_window", &F1TenthDPWSvgp::set_v_ref_window,
             py::arg("window"))
        .def("set_action_prior", &F1TenthDPWSvgp::set_action_prior,
             py::arg("mean"), py::arg("std"))
        .def("set_rollout_horizon", &F1TenthDPWSvgp::set_rollout_horizon,
             py::arg("horizon"))
        .def("set_centerline", &F1TenthDPWSvgp::set_centerline,
             py::arg("cx"), py::arg("cy"), py::arg("heading"))
        .def("learn", &F1TenthDPWSvgp::learn, py::arg("Nsim"))
        .def("learn_timed", &F1TenthDPWSvgp::learn_timed, py::arg("budget_s"))
        .def("best_action", &F1TenthDPWSvgp::best_action)
        .def("forward", &F1TenthDPWSvgp::forward,
             py::arg("action"), py::arg("new_state"))
        .def("set_state", &F1TenthDPWSvgp::set_state, py::arg("state"))
        .def("get_root_actions", &F1TenthDPWSvgp::get_root_actions)
        .def("get_best_trajectory", &F1TenthDPWSvgp::get_best_trajectory,
             py::arg("max_depth") = 50);
}
