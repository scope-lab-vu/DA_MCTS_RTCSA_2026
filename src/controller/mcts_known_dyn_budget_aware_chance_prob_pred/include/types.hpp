#pragma once

#include <array>

constexpr int STATE_DIM = 7;  // [x, y, delta, v, psi, omega, beta]
constexpr int ACTION_DIM = 2; // [accel, sv]

using State7 = std::array<double, STATE_DIM>;
using Action2 = std::array<double, ACTION_DIM>;
