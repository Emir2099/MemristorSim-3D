#include "Memristor.h"
#include <cmath>
#include <random>

PhysicsEngine::PhysicsEngine(const MemristorParams& p) : m_params(p), m_w(p.w_init), m_r(0.0), m_i(0.0), m_power(0.0) {
    std::random_device rd;
    m_rng.seed(rd());
}

void PhysicsEngine::reset() { m_w = m_params.w_init; }

static inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

double PhysicsEngine::dw_dt(double v, double w) const {
    if (v > m_params.v_off) return m_params.k_off * std::pow((v / m_params.v_off) - 1.0, m_params.alpha_off);
    if (v < m_params.v_on) return m_params.k_on * std::pow((v / m_params.v_on) - 1.0, m_params.alpha_on);
    return 0.0;
}

double PhysicsEngine::rk4(double dt, double v, double w0) const {
    double k1 = dw_dt(v, w0);
    double k2 = dw_dt(v, w0 + 0.5 * dt * k1);
    double k3 = dw_dt(v, w0 + 0.5 * dt * k2);
    double k4 = dw_dt(v, w0 + dt * k3);
    return w0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void PhysicsEngine::update(double dt, double voltage) {
    double w_new = rk4(dt, voltage, m_w);
    double r_on = m_params.R_on;
    double r_off = m_params.R_off;
    m_w = clamp01(w_new);
    m_r = r_on + (r_off - r_on) * (1.0 - m_w);
    double raw_i = voltage / m_r;
    if (raw_i > m_params.I_compliance) raw_i = m_params.I_compliance;
    if (raw_i < -m_params.I_compliance) raw_i = -m_params.I_compliance;
    double noise = m_norm(m_rng) * (0.05 * raw_i);
    m_i = raw_i + noise;
    m_power = std::fabs(m_i * voltage);
    double dT = m_power * m_params.theta_thermal;
    if (dT > m_params.T_critical) {
        m_w = clamp01(m_w - dt * std::abs(m_params.k_off));
        m_r = r_on + (r_off - r_on) * (1.0 - m_w);
        raw_i = voltage / m_r;
        if (raw_i > m_params.I_compliance) raw_i = m_params.I_compliance;
        if (raw_i < -m_params.I_compliance) raw_i = -m_params.I_compliance;
        noise = m_norm(m_rng) * (0.05 * raw_i);
        m_i = raw_i + noise;
        m_power = std::fabs(m_i * voltage);
    }
}

double PhysicsEngine::w() const { return m_w; }
double PhysicsEngine::r() const { return m_r; }
double PhysicsEngine::i() const { return m_i; }
double PhysicsEngine::power() const { return m_power; }
std::pair<double,double> PhysicsEngine::iv_point(double v) const { return {v, m_i}; }
MemristorParams& PhysicsEngine::params() { return m_params; }
void PhysicsEngine::set_params(const MemristorParams& p) { m_params = p; }
