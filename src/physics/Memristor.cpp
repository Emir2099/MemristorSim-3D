#include "Memristor.h"
#include <cmath>
#include <random>

PhysicsEngine::PhysicsEngine(const MemristorParams& p) : m_params(p), m_w(p.w_init), m_r(0.0), m_i(0.0), m_power(0.0), m_dT(0.0) {
    std::random_device rd;
    m_rng.seed(rd());
}

void PhysicsEngine::reset() { 
    m_w = m_params.w_init; 
    m_dT = 0.0;
}

static inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

double PhysicsEngine::get_dw_dt(double v, double w, double dT) const {
    double dw = 0.0;
    if (v > m_params.v_off) {
        // RESET process: trying to turn OFF (w -> 0.0)
        // k_off is negative, so this term will be negative
        dw = m_params.k_off * std::pow((v / m_params.v_off) - 1.0, m_params.alpha_off);
        // Biolek window for w decreasing towards 0
        dw *= (1.0 - std::pow(w - 1.0, 8.0));
    } else if (v < m_params.v_on) {
        // SET process: trying to turn ON (w -> 1.0)
        // k_on is positive, so this term will be positive
        dw = m_params.k_on * std::pow((v / m_params.v_on) - 1.0, m_params.alpha_on);
        // Biolek window for w increasing towards 1
        dw *= (1.0 - std::pow(w, 8.0));
    }
    
    // Smooth thermal dissolution: if temperature rise exceeds T_critical,
    // decay filament back to 0. Rate is proportional to excess temperature.
    if (dT > m_params.T_critical) {
        double thermal_decay = -std::abs(m_params.k_off) * ((dT - m_params.T_critical) / m_params.T_critical) * w;
        dw += thermal_decay;
    }
    
    return dw;
}

double PhysicsEngine::rk4(double dt, double v, double w0, double dT) const {
    double k1 = get_dw_dt(v, w0, dT);
    double k2 = get_dw_dt(v, w0 + 0.5 * dt * k1, dT);
    double k3 = get_dw_dt(v, w0 + 0.5 * dt * k2, dT);
    double k4 = get_dw_dt(v, w0 + dt * k3, dT);
    return w0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void PhysicsEngine::update(double dt, double voltage) {
    if (dt <= 0.0) return;
    
    // Integrate state variable w using RK4
    double w_new = rk4(dt, voltage, m_w, m_dT);
    m_w = clamp01(w_new);
    
    // State-based equivalent resistance
    double r_on = m_params.R_on;
    double r_off = m_params.R_off;
    m_r = r_on + (r_off - r_on) * (1.0 - m_w);
    
    // Calculate current using a highly realistic nonlinear conduction model
    // Ohmic in ON state (w=1), sinh-nonlinear in OFF state (w=0)
    double gamma = 2.0; // standard tunneling/Schottky nonlinearity factor
    double sinh_v = std::sinh(gamma * voltage);
    double sinh_1 = std::sinh(gamma);
    double i_on = voltage / r_on;
    double i_off = sinh_v / (r_off * sinh_1);
    double raw_i = m_w * i_on + (1.0 - m_w) * i_off;
    
    // Enforce current compliance limit
    if (raw_i > m_params.I_compliance) raw_i = m_params.I_compliance;
    if (raw_i < -m_params.I_compliance) raw_i = -m_params.I_compliance;
    
    // Add realistic current noise
    double noise = m_norm(m_rng) * (0.05 * raw_i);
    m_i = raw_i + noise;
    
    // Calculate instantaneous power dissipation
    m_power = std::fabs(m_i * voltage);
    
    // Solve dynamic heat equation (RC low-pass network) for temperature rise m_dT
    double tau_thermal = 0.01; 
    double dT_target = m_power * m_params.theta_thermal;
    m_dT += (dt / (dt + tau_thermal)) * (dT_target - m_dT);
}

double PhysicsEngine::w() const { return m_w; }
double PhysicsEngine::r() const { return m_r; }
double PhysicsEngine::i() const { return m_i; }
double PhysicsEngine::power() const { return m_power; }
double PhysicsEngine::dT() const { return m_dT; }
std::pair<double,double> PhysicsEngine::iv_point(double v) const { return {v, m_i}; }
MemristorParams& PhysicsEngine::params() { return m_params; }
void PhysicsEngine::set_params(const MemristorParams& p) { m_params = p; }
