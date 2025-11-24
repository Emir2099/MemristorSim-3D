#pragma once
#include <utility>
#include <random>

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

