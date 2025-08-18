
#include "gait_ref.hpp"
#include <algorithm>
#include <cmath>

static inline double Cnk(int n, int k) {
  double r = 1.0;
  for (int i = 1; i <= k; ++i) {
    r = r * (n + 1 - i) / i;
  }
  return r;
}

std::array<double, 2> Bezier2D::sample(double t) const {
  int n = (int)cp.size() - 1;
  double x = 0.0, z = 0.0;
  for (int i = 0; i <= n; ++i) {
    double b = Cnk(n, i) * std::pow(1 - t, n - i) * std::pow(t, i);
    x += b * cp[i][0];
    z += b * cp[i][1];
  }
  return {x, z};
}

std::vector<std::array<double, 2>>
Bezier2D::default_cp(double L_span, double base_h, double clearance) {
  std::vector<std::array<double, 2>> v(12);
  double xL = -L_span, xR = +L_span;
  double z0 = base_h, zs = base_h + clearance; // up(+)
  v[0] = {xL, z0};
  v[1] = {xL - 0.05, z0};
  v[2] = {xL - 0.06, zs};
  v[3] = {xL - 0.06, zs};
  v[4] = {xL - 0.06, zs};
  v[5] = {0.0, zs};
  v[6] = {0.0, zs};
  v[7] = {0.0, zs - 0.02};
  v[8] = {xR + 0.06, zs - 0.02};
  v[9] = {xR + 0.06, zs - 0.02};
  v[10] = {xR + 0.04, z0};
  v[11] = {xR, z0};
  return v;
}

std::pair<bool, double> TrotPlanner::phase(double time, int leg) const {
  double leg_phase = (leg == 0 || leg == 3) ? 0.0 : 0.5; // FL/RR in-phase
  double T = stride_T();
  double base = std::fmod(time + leg_phase * T, T);
  if (base < 0)
    base += T;
  if (base <= T_stance)
    return {false, base / T_stance};
  return {true, (base - T_stance) / T_swing};
}

bool IK2Link::solve(double x, double z, double &hip, double &knee) const {
  double r2 = x * x + z * z;
  if (r2 < 1e-9)
    r2 = 1e-9;
  double cK = clampd((L1 * L1 + L2 * L2 - r2) / (2 * L1 * L2), -1.0, 1.0);
  knee = M_PI - std::acos(cK);
  double phi = std::atan2(z, x);
  double sK = std::sin(knee), cK2 = std::cos(knee);
  double beta = std::atan2(L2 * sK, L1 + L2 * cK2);
  hip = phi - beta;
  return std::isfinite(hip) && std::isfinite(knee);
}

static inline void clampReach(double &x, double &z, double L1, double L2) {
  double r = std::sqrt(x * x + z * z);
  double rmax = 0.98 * (L1 + L2);
  if (r > rmax) {
    double s = rmax / std::max(1e-9, r);
    x *= s;
    z *= s;
  }
}

void LocomotionRef::set_speed(double vx) {
  vx_cmd = std::max(0.05, std::abs(vx));
  gait.T_stance = 2.0 * L_span / vx_cmd;
  gait.T_swing = std::max(0.15, 0.4 * gait.T_stance);
  swing_curve.cp = Bezier2D::default_cp(L_span, base_h, clearance);
}

void LocomotionRef::build_qref(std::vector<double> &qref,
                               const std::vector<int> &,
                               const std::vector<int> &, const mjModel *,
                               const mjData *) {
  qref.assign(qref.size(), 0.0);
  auto build_leg = [&](const LegIndex &leg, int leg_id) {
    if (leg.hip < 0 || leg.knee < 0)
      return;
    auto st = gait.phase(t, leg_id);
    bool swing = st.first;
    double s = std::min(1.0, std::max(0.0, st.second));
    double x = 0.0, z_up = base_h; // up(+)
    if (swing) {
      auto p = swing_curve.sample(s);
      x = p[0];
      z_up = p[1];
    } else {
      x = +L_span - 2.0 * L_span * s;
      z_up = base_h;
    }
    clampReach(x, z_up, ik.L1, ik.L2);
    double hip = 0.0, knee = 0.0;
    if (ik.solve(x, z_up, hip, knee)) {
      qref[leg.hip] = hip;
      qref[leg.knee] = knee;
    }
  };
  build_leg(fl, 0);
  build_leg(fr, 1);
  build_leg(rl, 2);
  build_leg(rr, 3);
}
