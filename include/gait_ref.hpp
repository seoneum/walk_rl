
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>
#include <vector>

struct Bezier2D {
  std::vector<std::array<double, 2>> cp; // {x,z} in meters
  std::array<double, 2> sample(double t) const;
  static std::vector<std::array<double, 2>>
  default_cp(double L_span, double base_h, double clearance);
};

struct TrotPlanner {
  double T_stance{0.3};
  double T_swing{0.15};
  double stride_T() const { return T_stance + T_swing; }
  std::pair<bool, double> phase(double time,
                                int leg) const; // (isSwing, phase 0..1)
};

struct IK2Link {
  double L1{0.10}, L2{0.10};
  static inline double clampd(double x, double a, double b) {
    return std::max(a, std::min(b, x));
  }
  bool solve(double x, double z, double &hip,
             double &knee) const; // input: x forward(+), z up(+)
};

struct LegIndex {
  int hip = -1, knee = -1, ankle = -1;
};

struct LocomotionRef {
  double L_span = 0.12, base_h = 0.18,
         clearance = 0.03; // base_h, clearance are positive (up)
  double vx_cmd = 0.4;
  IK2Link ik;
  TrotPlanner gait;
  Bezier2D swing_curve;
  double t = 0.0;

  LegIndex fl, fr, rl, rr;

  void set_speed(double vx);
  void step(double dt) { t += dt; }
  void build_qref(std::vector<double> &qref,
                  const std::vector<int> &act_qposadr,
                  const std::vector<int> &act_jnt_ids, const mjModel *m,
                  const mjData *d);
};
