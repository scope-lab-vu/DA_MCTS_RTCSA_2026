#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mcts_native.hpp"
#include "f1tenth_simulator.hpp"
#include <memory>

namespace py = pybind11;

// Numpy array -> State7
inline State7 to_state7(py::array_t<double> arr) {
    State7 s;
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(ptr, ptr + STATE_DIM, s.begin());
    return s;
}

// Numpy array -> Action2
inline Action2 to_action2(py::array_t<double> arr) {
    Action2 a;
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::copy(ptr, ptr + ACTION_DIM, a.begin());
    return a;
}

// State7 -> numpy array
inline py::array_t<double> to_numpy_state(const State7& s) {
    return py::array_t<double>(STATE_DIM, s.data());
}

// Action2 -> numpy array
inline py::array_t<double> to_numpy_action(const Action2& a) {
    return py::array_t<double>(ACTION_DIM, a.data());
}

// Numpy array -> std::vector<double> (for variable-length data)
inline std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    double* ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

// Numpy float32 array -> std::vector<float>
inline std::vector<float> to_float_vec(py::array_t<float> arr) {
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    return std::vector<float>(ptr, ptr + buf.size);
}

// Wrapper class that owns both the simulator and the DPW tree.
// Exposed to Python as a single object with a clean API.
class F1TenthMCTSSvgpBudgetProbPred {
    std::unique_ptr<F1TenthSimulator> sim_;
    std::unique_ptr<DPWNative> model_;

  public:
    F1TenthMCTSSvgpBudgetProbPred(py::array_t<double> initial_obs,
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
               // Krasser SVGP params (two flat arrays: omega + beta)
               py::array_t<double> omega_data,
               py::array_t<double> beta_data,
               int svgp_M_omega, int svgp_M_beta,
               int svgp_D_input,
               double base_timestep,
               unsigned seed = 0) {

        auto obs = to_state7(initial_obs);

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

        // Build Krasser SVGP params from flat arrays packed in Python
        auto omega_buf = omega_data.request();
        auto beta_buf  = beta_data.request();
        auto omega_svgp = std::make_shared<KrasserSVGPParams>(
            KrasserSVGPParams::from_flat(static_cast<double*>(omega_buf.ptr),
                                         svgp_M_omega, svgp_D_input));
        auto beta_svgp = std::make_shared<KrasserSVGPParams>(
            KrasserSVGPParams::from_flat(static_cast<double*>(beta_buf.ptr),
                                         svgp_M_beta, svgp_D_input));

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

    void set_is_curve(bool is_curve) {
        sim_->set_is_curve(is_curve);
    }

    void set_centerline(py::array_t<double> cx, py::array_t<double> cy,
                        py::array_t<double> heading) {
        sim_->set_centerline(to_vec(cx), to_vec(cy), to_vec(heading));
    }

    void set_c_alpha(double c_alpha) { sim_->set_c_alpha(c_alpha); }
    void set_no_c_alpha(bool flag) { sim_->set_no_c_alpha(flag); }

    void set_budget_params(double W_low, double W_high, int N_w, int N_r) {
        model_->set_budget_params(W_low, W_high, N_w, N_r);
    }

    void set_action_weights(double p_agg, double p_slow, double p_feas) {
        sim_->set_action_weights(p_agg, p_slow, p_feas);
    }

    // Load MLP weights from Python numpy float32 arrays.
    // Call once after construction (before first learn_timed()).
    void load_mlp_weights(py::array_t<float> W0, py::array_t<float> b0,
                          py::array_t<float> W1, py::array_t<float> b1,
                          py::array_t<float> W2, py::array_t<float> b2,
                          py::array_t<float> mu, py::array_t<float> sigma) {
        auto to_fvec = [](py::array_t<float> a) {
            auto buf = a.request();
            float* ptr = static_cast<float*>(buf.ptr);
            return std::vector<float>(ptr, ptr + buf.size);
        };
        model_->load_mlp_weights(to_fvec(W0), to_fvec(b0),
                                  to_fvec(W1), to_fvec(b1),
                                  to_fvec(W2), to_fvec(b2),
                                  to_fvec(mu), to_fvec(sigma));
    }

    void set_context(double v, double kappa) { model_->set_context(v, kappa); }

    void open_log(const std::string& path) { model_->open_log(path); }

    double get_last_tau() { return model_->get_last_tau(); }

    py::tuple get_last_weights() {
        auto w = model_->get_last_weights();
        return py::make_tuple(w[0], w[1], w[2]);
    }

    void learn(int Nsim) { model_->learn(Nsim); }

    int learn_timed(double budget_s) { return model_->learn_timed(budget_s); }

    int learn_n_mlp(int N_total) { return model_->learn_n_mlp(N_total); }

    py::array_t<double> best_action() {
        return to_numpy_action(model_->best_action_vec());
    }

    void forward(py::array_t<double> action, py::array_t<double> new_state) {
        model_->forward(to_action2(action), to_state7(new_state));
        // Reset simulator to the new root state
        sim_->set_state(to_state7(new_state));
    }

    void set_state(py::array_t<double> state) {
        sim_->set_state(to_state7(state));
    }

    // Return root actions as list of dicts: [{action, visits, avg_reward}, ...]
    py::list get_root_actions() {
        auto infos = model_->get_root_actions();
        py::list result;
        for (auto& info : infos) {
            py::dict d;
            d["action"] = to_numpy_action(info.action);
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
            d["state"] = to_numpy_state(step.state);
            d["action"] = to_numpy_action(step.action);
            d["avg_reward"] = step.avg_reward;
            d["visits"] = step.visits;
            result.append(d);
        }
        return result;
    }
};

PYBIND11_MODULE(mcts_svgp_cpp_budget_aware_chance_prob_pred, m) {
    m.doc() = "C++ MCTS DPW + Krasser SVGP + budget-aware action weighting with MLP probability prediction";

    py::class_<F1TenthMCTSSvgpBudgetProbPred>(m, "F1TenthMCTSSvgpBudgetProbPred")
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
                       // Krasser SVGP params
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
             // Krasser SVGP params
             py::arg("omega_data"), py::arg("beta_data"),
             py::arg("svgp_M_omega"), py::arg("svgp_M_beta"),
             py::arg("svgp_D_input"),
             py::arg("base_timestep"),
             py::arg("seed") = 0)
        .def("set_v_ref_window", &F1TenthMCTSSvgpBudgetProbPred::set_v_ref_window,
             py::arg("window"))
        .def("set_action_prior", &F1TenthMCTSSvgpBudgetProbPred::set_action_prior,
             py::arg("mean"), py::arg("std"))
        .def("set_rollout_horizon", &F1TenthMCTSSvgpBudgetProbPred::set_rollout_horizon,
             py::arg("horizon"))
        .def("set_is_curve", &F1TenthMCTSSvgpBudgetProbPred::set_is_curve,
             py::arg("is_curve"))
        .def("set_centerline", &F1TenthMCTSSvgpBudgetProbPred::set_centerline,
             py::arg("cx"), py::arg("cy"), py::arg("heading"))
        .def("set_c_alpha", &F1TenthMCTSSvgpBudgetProbPred::set_c_alpha,
             py::arg("c_alpha"))
        .def("set_no_c_alpha", &F1TenthMCTSSvgpBudgetProbPred::set_no_c_alpha,
             py::arg("flag"))
        .def("set_budget_params", &F1TenthMCTSSvgpBudgetProbPred::set_budget_params,
             py::arg("W_low"), py::arg("W_high"), py::arg("N_w"), py::arg("N_r"))
        .def("set_action_weights", &F1TenthMCTSSvgpBudgetProbPred::set_action_weights,
             py::arg("p_agg"), py::arg("p_slow"), py::arg("p_feas"))
        .def("load_mlp_weights", &F1TenthMCTSSvgpBudgetProbPred::load_mlp_weights,
             py::arg("W0"), py::arg("b0"),
             py::arg("W1"), py::arg("b1"),
             py::arg("W2"), py::arg("b2"),
             py::arg("mu"), py::arg("sigma"))
        .def("set_context", &F1TenthMCTSSvgpBudgetProbPred::set_context,
             py::arg("v"), py::arg("kappa"))
        .def("open_log", &F1TenthMCTSSvgpBudgetProbPred::open_log,
             py::arg("path"))
        .def("get_last_tau", &F1TenthMCTSSvgpBudgetProbPred::get_last_tau)
        .def("get_last_weights", &F1TenthMCTSSvgpBudgetProbPred::get_last_weights)
        .def("learn", &F1TenthMCTSSvgpBudgetProbPred::learn, py::arg("Nsim"))
        .def("learn_timed", &F1TenthMCTSSvgpBudgetProbPred::learn_timed, py::arg("budget_s"))
        .def("learn_n_mlp", &F1TenthMCTSSvgpBudgetProbPred::learn_n_mlp, py::arg("N_total"))
        .def("best_action", &F1TenthMCTSSvgpBudgetProbPred::best_action)
        .def("forward", &F1TenthMCTSSvgpBudgetProbPred::forward,
             py::arg("action"), py::arg("new_state"))
        .def("set_state", &F1TenthMCTSSvgpBudgetProbPred::set_state, py::arg("state"))
        .def("get_root_actions", &F1TenthMCTSSvgpBudgetProbPred::get_root_actions)
        .def("get_best_trajectory", &F1TenthMCTSSvgpBudgetProbPred::get_best_trajectory,
             py::arg("max_depth") = 50);
}
