#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mcts_native.hpp"
#include "f1tenth_simulator.hpp"

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

// Convert flat float32 array (H*F,) -> 2D vector [H][F]
inline std::vector<std::vector<float>> to_2d_float(
    py::array_t<float> arr, int rows, int cols
) {
    auto flat = to_float_vec(arr);
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            result[r][c] = flat[r * cols + c];
    return result;
}

// Wrapper class that owns both the simulator and the DPW tree.
// Exposed to Python as a single object with a clean API.
class F1TenthDPW {
    std::unique_ptr<F1TenthSimulator> sim_;
    std::unique_ptr<DPWNative> model_;

  public:
    F1TenthDPW(py::array_t<double> initial_obs,
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
               // Occupancy map
               py::array_t<float> occ_grid,
               double occ_x_min, double occ_y_min,
               double occ_cell_size, int occ_nx, int occ_ny,
               // Cost params
               double speed_weight, double alpha_threshold,
               double v_ref, int rollout_horizon,
               double discount, double dt,
               // PINN params
               const std::string& model_path,
               py::array_t<float> scaler_mean,
               py::array_t<float> scaler_scale,
               py::array_t<float> feature_history) {

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

        CostParams cost;
        cost.speed_weight    = speed_weight;
        cost.alpha_threshold = alpha_threshold;
        cost.v_ref           = v_ref;
        cost.rollout_horizon = rollout_horizon;
        cost.discount_factor = discount;
        cost.dt              = dt;

        auto mean_vec  = to_float_vec(scaler_mean);
        auto scale_vec = to_float_vec(scaler_scale);

        sim_ = std::make_unique<F1TenthSimulator>(
            vp, occ, cost, model_path, mean_vec, scale_vec);
        sim_->set_state(obs);

        // Set initial feature history (10x6, passed as flat array)
        auto hist = to_2d_float(feature_history, 10, 6);
        sim_->set_feature_history(hist);

        model_ = std::make_unique<DPWNative>(sim_.get(), obs, alpha, beta, K);
    }

    void set_v_ref_window(py::array_t<double> window) {
        sim_->set_v_ref_window(to_vec(window));
    }

    void set_feature_history(py::array_t<float> history) {
        auto hist = to_2d_float(history, 10, 6);
        sim_->set_feature_history(hist);
    }

    void learn(int Nsim) { model_->learn(Nsim); }

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
};

PYBIND11_MODULE(mcts_f1tenth_pinn_cpp, m) {
    m.doc() = "C++ MCTS DPW for F1TENTH racing with PINN dynamics (LibTorch)";

    py::class_<F1TenthDPW>(m, "F1TenthDPW")
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
                       // Occupancy map
                       py::array_t<float>,
                       double, double, double, int, int,
                       // Cost params
                       double, double, double, int, double, double,
                       // PINN params
                       const std::string&,
                       py::array_t<float>,
                       py::array_t<float>,
                       py::array_t<float>>(),
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
             // Occupancy map
             py::arg("occ_grid"),
             py::arg("occ_x_min"), py::arg("occ_y_min"),
             py::arg("occ_cell_size"), py::arg("occ_nx"), py::arg("occ_ny"),
             // Cost params
             py::arg("speed_weight"), py::arg("alpha_threshold"),
             py::arg("v_ref"), py::arg("rollout_horizon"),
             py::arg("discount"), py::arg("dt"),
             // PINN params
             py::arg("model_path"),
             py::arg("scaler_mean"),
             py::arg("scaler_scale"),
             py::arg("feature_history"))
        .def("set_v_ref_window", &F1TenthDPW::set_v_ref_window,
             py::arg("window"))
        .def("set_feature_history", &F1TenthDPW::set_feature_history,
             py::arg("history"))
        .def("learn", &F1TenthDPW::learn, py::arg("Nsim"))
        .def("best_action", &F1TenthDPW::best_action)
        .def("forward", &F1TenthDPW::forward,
             py::arg("action"), py::arg("new_state"))
        .def("set_state", &F1TenthDPW::set_state, py::arg("state"));
}
