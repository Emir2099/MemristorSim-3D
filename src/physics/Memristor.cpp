#include "Memristor.h"
#include <cmath>
#include <random>

PhysicsEngine::PhysicsEngine(const MemristorParams& p) 
    : m_params(p), m_active_params(p), m_w(p.w_init), m_r(0.0), m_i(0.0), m_power(0.0), m_dT(0.0), m_rtn_state(0) {
    std::random_device rd;
    m_rng.seed(rd());
    apply_d2d_variability();
    m_w = m_active_params.w_init;
}

void PhysicsEngine::reset() { 
    apply_d2d_variability();
    m_w = m_active_params.w_init; 
    m_dT = 0.0;
    m_rtn_state = 0;
}

static inline double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

void PhysicsEngine::apply_d2d_variability() {
    m_active_params = m_params;
    if (m_params.enable_variability) {
        // D2D w_init: Normal distribution
        double w_var = m_norm(m_rng) * m_params.sigma_w_init;
        m_active_params.w_init = clamp01(m_params.w_init + w_var);

        // D2D k_on, k_off: Log-normal distribution (exponential barrier changes)
        double log_k_on_var = m_norm(m_rng) * m_params.sigma_k_on;
        double log_k_off_var = m_norm(m_rng) * m_params.sigma_k_on;
        m_active_params.k_on = m_params.k_on * std::pow(10.0, log_k_on_var);
        m_active_params.k_off = m_params.k_off * std::pow(10.0, log_k_off_var);
    }
}

double PhysicsEngine::get_dw_dt(double v, double w, double dT) const {
    double dw = 0.0;
    if (v > m_active_params.v_off) {
        // RESET process: trying to turn OFF (w -> 0.0)
        // k_off is negative, so this term will be negative
        dw = m_active_params.k_off * std::pow((v / m_active_params.v_off) - 1.0, m_active_params.alpha_off);
        // Biolek window for w decreasing towards 0
        dw *= (1.0 - std::pow(w - 1.0, 8.0));
    } else if (v < m_active_params.v_on) {
        // SET process: trying to turn ON (w -> 1.0)
        // k_on is positive, so this term will be positive
        dw = m_active_params.k_on * std::pow((v / m_active_params.v_on) - 1.0, m_active_params.alpha_on);
        // Biolek window for w increasing towards 1
        dw *= (1.0 - std::pow(w, 8.0));
    }
    
    // Smooth thermal dissolution: if temperature rise exceeds T_critical,
    // decay filament back to 0. Rate is proportional to excess temperature.
    if (dT > m_active_params.T_critical) {
        double thermal_decay = -std::abs(m_active_params.k_off) * ((dT - m_active_params.T_critical) / m_active_params.T_critical) * w;
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
    
    // Solve for the voltage across the memristor component (1S1R / 1T1R series drop)
    double v_mem = voltage;
    if (m_active_params.enable_selector) {
        double low = (voltage > 0.0) ? 0.0 : voltage;
        double high = (voltage > 0.0) ? voltage : 0.0;
        for (int iter = 0; iter < 12; ++iter) {
            double mid = (low + high) * 0.5;
            double i_mem = calculate_memristor_current(mid);
            double i_sel = calculate_selector_current(voltage - mid);
            if (i_mem > i_sel) {
                if (voltage > 0.0) high = mid; else low = mid;
            } else {
                if (voltage > 0.0) low = mid; else high = mid;
            }
        }
        v_mem = (low + high) * 0.5;
    }
    
    // Integrate state variable w using RK4 based on the actual voltage across the memristor
    double w_new = rk4(dt, v_mem, m_w, m_dT);
    
    // Apply C2C write noise (stochastic SDE term: sigma * sqrt(dt) * N(0, 1))
    if (m_active_params.enable_variability) {
        double c2c_noise = m_active_params.sigma_c2c * std::sqrt(dt) * m_norm(m_rng);
        w_new += c2c_noise;
    }
    m_w = clamp01(w_new);
    
    // State-based equivalent resistance
    double r_on = m_active_params.R_on;
    double r_off = m_active_params.R_off;
    m_r = r_on + (r_off - r_on) * (1.0 - m_w);
    
    // RTN state update first
    if (m_active_params.enable_rtn) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r_val = dist(m_rng);
        if (m_rtn_state == 0) {
            double p_transition = 1.0 - std::exp(-dt / m_active_params.rtn_tau_c);
            if (r_val < p_transition) m_rtn_state = 1;
        } else {
            double p_transition = 1.0 - std::exp(-dt / m_active_params.rtn_tau_e);
            if (r_val < p_transition) m_rtn_state = 0;
        }
    }
    
    // Calculate raw current using the modular method
    double raw_i = calculate_current(voltage);
    
    // Add realistic read thermal current noise (5% SD)
    double noise = m_norm(m_rng) * (0.05 * raw_i);
    m_i = raw_i + noise;
    
    // Calculate instantaneous power dissipation
    m_power = std::fabs(m_i * voltage);
    
    // Solve dynamic heat equation
    double tau_thermal = 0.01; 
    double dT_target = m_power * m_active_params.theta_thermal;
    m_dT += (dt / (dt + tau_thermal)) * (dT_target - m_dT);
}

double PhysicsEngine::w() const { return m_w; }
double PhysicsEngine::r() const { return m_r; }
double PhysicsEngine::i() const { return m_i; }
double PhysicsEngine::power() const { return m_power; }
double PhysicsEngine::dT() const { return m_dT; }
std::pair<double,double> PhysicsEngine::iv_point(double v) const { return {v, m_i}; }
MemristorParams& PhysicsEngine::params() { return m_params; }
void PhysicsEngine::set_params(const MemristorParams& p) { 
    m_params = p; 
    apply_d2d_variability();
}
void PhysicsEngine::set_w(double w) {
    m_w = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
    double r_on = m_active_params.R_on;
    double r_off = m_active_params.R_off;
    m_r = r_on + (r_off - r_on) * (1.0 - m_w);
}

double PhysicsEngine::calculate_memristor_current(double voltage_diff) const {
    double r_on = m_active_params.R_on;
    double r_off = m_active_params.R_off;
    
    // Calculate current using a highly realistic nonlinear conduction model
    // Ohmic in ON state (w=1), and selectable nonlinear in OFF state (w=0)
    double i_on = voltage_diff / r_on;
    double i_off = 0.0;
    double abs_v = std::fabs(voltage_diff);
    double sgn_v = (voltage_diff > 0.0) ? 1.0 : ((voltage_diff < 0.0) ? -1.0 : 0.0);
    
    if (m_active_params.conduction_model == ConductionModel::Sinh) {
        double gamma = m_active_params.gamma_sinh;
        double sinh_v = std::sinh(gamma * voltage_diff);
        double sinh_1 = std::sinh(gamma);
        i_off = sinh_v / (r_off * sinh_1);
    } else if (m_active_params.conduction_model == ConductionModel::PooleFrenkel) {
        // Poole-Frenkel Emission: ln(I/V) is proportional to sqrt(V)
        i_off = (voltage_diff / r_off) * std::exp(m_active_params.beta_pf * (std::sqrt(abs_v) - 1.0));
    } else if (m_active_params.conduction_model == ConductionModel::Schottky) {
        // Schottky Tunneling / Emission: ln(I) is proportional to sqrt(V)
        i_off = (sgn_v / r_off) * std::exp(m_active_params.beta_sc * (std::sqrt(abs_v) - 1.0));
    }
    
    double raw_i = m_w * i_on + (1.0 - m_w) * i_off;
    
    // RTN Simulation relative current fluctuation
    if (m_active_params.enable_rtn) {
        double rtn_factor = 1.0 + (m_rtn_state == 1 ? 0.5 : -0.5) * m_active_params.rtn_amplitude;
        raw_i *= rtn_factor;
    }
    
    // Enforce current compliance limit
    if (raw_i > m_active_params.I_compliance) raw_i = m_active_params.I_compliance;
    if (raw_i < -m_active_params.I_compliance) raw_i = -m_active_params.I_compliance;
    
    return raw_i;
}

double PhysicsEngine::calculate_selector_current(double v_sel) const {
    double abs_v = std::fabs(v_sel);
    double sgn_v = (v_sel > 0.0) ? 1.0 : ((v_sel < 0.0) ? -1.0 : 0.0);
    
    if (m_active_params.selector_type == 0) {
        // 1S1R Volatile Threshold Switch Model
        // G_off is 1e-9 S (1 GOhm) to eliminate leakage, G_on is 1e-3 S (1 kOhm)
        double g_off = 1e-9;
        double g_on = 1e-3;
        double v_th = m_active_params.selector_v_th;
        
        // Smooth transition representing volatile threshold switching
        double conduct = g_off + (g_on - g_off) / (1.0 + std::exp(- (abs_v - v_th) / 0.05));
        return v_sel * conduct;
    } else {
        // 1T1R Transistor Selector Model (Square-law MOSFET model)
        double v_gate = m_active_params.selector_v_gate;
        double v_th_trans = m_active_params.selector_v_th_trans;
        double beta = 2.0e-3; // Transconductance beta (A/V^2)
        
        double v_overdrive = v_gate - v_th_trans;
        if (v_overdrive <= 0.0) return 0.0;
        
        if (abs_v < v_overdrive) {
            // Linear region
            return sgn_v * beta * (v_overdrive * abs_v - 0.5 * abs_v * abs_v);
        } else {
            // Saturation region
            return sgn_v * 0.5 * beta * v_overdrive * v_overdrive;
        }
    }
}

double PhysicsEngine::calculate_current(double voltage_diff) const {
    if (!m_active_params.enable_selector) {
        return calculate_memristor_current(voltage_diff);
    }
    
    double low = (voltage_diff > 0.0) ? 0.0 : voltage_diff;
    double high = (voltage_diff > 0.0) ? voltage_diff : 0.0;
    for (int iter = 0; iter < 12; ++iter) {
        double mid = (low + high) * 0.5;
        double i_mem = calculate_memristor_current(mid);
        double i_sel = calculate_selector_current(voltage_diff - mid);
        if (i_mem > i_sel) {
            if (voltage_diff > 0.0) high = mid; else low = mid;
        } else {
            if (voltage_diff > 0.0) low = mid; else high = mid;
        }
    }
    double v_mem = (low + high) * 0.5;
    return calculate_memristor_current(v_mem);
}
