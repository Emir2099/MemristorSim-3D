#pragma once
#include <utility>
#include <random>
#include <string>
#include <map>

enum class ConductionModel { Sinh, PooleFrenkel, Schottky };

struct MemristorParams {
    double v_off = 1.0;
    double v_on = -1.0;
    double k_off = -100.0;
    double k_on = 100.0;
    double alpha_off = 3.0;
    double alpha_on = 3.0;
    double R_off = 10000.0;
    double R_on = 100.0;
    double w_init = 0.0;
    double theta_thermal = 0.01;
    double T_critical = 5.0;
    double I_compliance = 0.05;
    
    // Advanced Conduction Models
    ConductionModel conduction_model = ConductionModel::Sinh;
    double gamma_sinh = 2.0;
    double beta_pf = 1.5;
    double beta_sc = 2.0;

    // Stochasticity & RTN Parameters
    bool enable_variability = false;
    double sigma_w_init = 0.05;      // D2D variability of w_init (Normal)
    double sigma_k_on = 0.15;        // D2D variability of k_on/k_off (Log-normal factor)
    double sigma_c2c = 0.02;         // C2C write noise rate (SDE diffusion coefficient)

    bool enable_rtn = false;
    double rtn_amplitude = 0.03;     // RTN relative current fluctuation (e.g., 3%)
    double rtn_tau_c = 0.05;         // Mean capture time (s)
    double rtn_tau_e = 0.05;         // Mean emission time (s)
    
    // Selector Device Parameters (1S1R / 1T1R)
    bool enable_selector = false;
    int selector_type = 0;           // 0 = 1S1R (Threshold Switch), 1 = 1T1R (Transistor)
    double selector_v_th = 0.6;      // Selector Threshold Voltage (V)
    double selector_alpha = 10.0;    // Selector slope/nonlinearity
    double selector_v_gate = 1.8;    // 1T1R Gate Voltage (V)
    double selector_v_th_trans = 0.4; // 1T1R Transistor threshold voltage (V)
};

class PhysicsEngine {
public:
    explicit PhysicsEngine(const MemristorParams& p);
    void reset();
    void update(double dt, double voltage);
    double w() const;
    double r() const;
    double i() const;
    double power() const;
    double dT() const;
    std::pair<double,double> iv_point(double v) const;
    MemristorParams& params();
    void set_params(const MemristorParams& p);
    void set_w(double w);
    double calculate_current(double voltage_diff) const;
    double calculate_memristor_current(double voltage_diff) const;
    double calculate_selector_current(double v_sel) const;
private:
    MemristorParams m_params;
    MemristorParams m_active_params;
    double m_w;
    double m_r;
    double m_i;
    double m_power;
    double m_dT;
    int m_rtn_state = 0;
    std::default_random_engine m_rng;
    std::normal_distribution<double> m_norm{0.0, 1.0};
    double get_dw_dt(double v, double w, double dT) const;
    double rk4(double dt, double v, double w0, double dT) const;
    void apply_d2d_variability();
};

struct MaterialPreset {
    std::string name;
    MemristorParams params;
};

class MemristorLibrary {
public:
    static std::map<std::string, MemristorParams> GetPresets() {
        std::map<std::string, MemristorParams> presets;
        MemristorParams ag_si; ag_si.v_on = -1.2; ag_si.v_off = 1.0; ag_si.k_on = 1000.0; ag_si.k_off = -1000.0; ag_si.alpha_on = 3.0; ag_si.alpha_off = 3.0; ag_si.R_on = 50.0; ag_si.R_off = 15000.0; presets["Ag/a-Si (Fast)"] = ag_si;
        MemristorParams hfox; hfox.v_on = -0.8; hfox.v_off = 0.8; hfox.k_on = 100.0; hfox.k_off = -100.0; hfox.alpha_on = 4.0; hfox.alpha_off = 4.0; hfox.R_on = 100.0; hfox.R_off = 20000.0; presets["HfOx (Standard)"] = hfox;
        MemristorParams linear; linear.v_on = -0.1; linear.v_off = 0.1; linear.k_on = 10.0; linear.k_off = -10.0; linear.alpha_on = 1.0; linear.alpha_off = 1.0; linear.R_on = 100.0; linear.R_off = 5000.0; presets["Linear Drift (Edu)"] = linear;
        return presets;
    }
};
