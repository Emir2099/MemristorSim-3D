#pragma once
#include <utility>
#include <random>
#include <string>
#include <map>

struct MemristorParams {
    double v_off = 1.0;
    double v_on = -1.0;
    double k_off = 100.0;
    double k_on = -100.0;
    double alpha_off = 3.0;
    double alpha_on = 3.0;
    double R_off = 10000.0;
    double R_on = 100.0;
    double w_init = 0.0;
    double theta_thermal = 0.01;
    double T_critical = 5.0;
    double I_compliance = 0.05;
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
    std::pair<double,double> iv_point(double v) const;
    MemristorParams& params();
    void set_params(const MemristorParams& p);
private:
    MemristorParams m_params;
    double m_w;
    double m_r;
    double m_i;
    double m_power;
    std::default_random_engine m_rng;
    std::normal_distribution<double> m_norm{0.0, 1.0};
    double dw_dt(double v, double w) const;
    double rk4(double dt, double v, double w0) const;
};

struct MaterialPreset {
    std::string name;
    MemristorParams params;
};

class MemristorLibrary {
public:
    static std::map<std::string, MemristorParams> GetPresets() {
        std::map<std::string, MemristorParams> presets;
        MemristorParams ag_si; ag_si.v_on = -1.2; ag_si.v_off = 1.0; ag_si.k_on = -1000.0; ag_si.k_off = 1000.0; ag_si.alpha_on = 3.0; ag_si.alpha_off = 3.0; ag_si.R_on = 50.0; ag_si.R_off = 15000.0; presets["Ag/a-Si (Fast)"] = ag_si;
        MemristorParams hfox; hfox.v_on = -0.8; hfox.v_off = 0.8; hfox.k_on = -100.0; hfox.k_off = 100.0; hfox.alpha_on = 4.0; hfox.alpha_off = 4.0; hfox.R_on = 100.0; hfox.R_off = 20000.0; presets["HfOx (Standard)"] = hfox;
        MemristorParams linear; linear.v_on = -0.1; linear.v_off = 0.1; linear.k_on = -10.0; linear.k_off = 10.0; linear.alpha_on = 1.0; linear.alpha_off = 1.0; linear.R_on = 100.0; linear.R_off = 5000.0; presets["Linear Drift (Edu)"] = linear;
        return presets;
    }
};
