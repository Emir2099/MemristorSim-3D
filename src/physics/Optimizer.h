#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>
#include "Memristor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct FitDataPoint {
    double t;
    double v;
    double i;
};

class NelderMead {
public:
    using CostFunc = std::function<double(const std::vector<double>&)>;

    static std::vector<double> Optimize(CostFunc cost, std::vector<double> init, double tol = 1e-4, int max_iter = 300) {
        int n = init.size();
        std::vector<std::vector<double>> simplex(n + 1, std::vector<double>(n));
        std::vector<double> f(n + 1);

        simplex[0] = init;
        f[0] = cost(init);

        for (int i = 1; i <= n; ++i) {
            std::vector<double> point = init;
            point[i - 1] += (std::abs(point[i - 1]) > 0.0) ? point[i - 1] * 0.15 : 0.15;
            simplex[i] = point;
            f[i] = cost(point);
        }

        double alpha = 1.0;
        double gamma = 2.0;
        double rho = 0.5;
        double sigma = 0.5;

        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<int> idx(n + 1);
            for (int i = 0; i <= n; ++i) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return f[a] < f[b]; });

            std::vector<std::vector<double>> sorted_simplex(n + 1);
            std::vector<double> sorted_f(n + 1);
            for (int i = 0; i <= n; ++i) {
                sorted_simplex[i] = simplex[idx[i]];
                sorted_f[i] = f[idx[i]];
            }
            simplex = sorted_simplex;
            f = sorted_f;

            double diff = 0.0;
            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < n; ++j) {
                    diff = std::max(diff, std::abs(simplex[i][j] - simplex[0][j]));
                }
            }
            if (diff < tol) break;

            std::vector<double> centroid(n, 0.0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    centroid[j] += simplex[i][j];
                }
            }
            for (int j = 0; j < n; ++j) centroid[j] /= n;

            std::vector<double> xr(n);
            for (int j = 0; j < n; ++j) {
                xr[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j]);
            }
            double fxr = cost(xr);

            if (f[0] <= fxr && fxr < f[n - 1]) {
                simplex[n] = xr;
                f[n] = fxr;
                continue;
            }

            if (fxr < f[0]) {
                std::vector<double> xe(n);
                for (int j = 0; j < n; ++j) {
                    xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
                }
                double fxe = cost(xe);
                if (fxe < fxr) {
                    simplex[n] = xe;
                    f[n] = fxe;
                } else {
                    simplex[n] = xr;
                    f[n] = fxr;
                }
                continue;
            }

            if (fxr >= f[n - 1]) {
                std::vector<double> xc(n);
                if (fxr < f[n]) {
                    for (int j = 0; j < n; ++j) {
                        xc[j] = centroid[j] + rho * (xr[j] - centroid[j]);
                    }
                    double fxc = cost(xc);
                    if (fxc < fxr) {
                        simplex[n] = xc;
                        f[n] = fxc;
                        continue;
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        xc[j] = centroid[j] - rho * (centroid[j] - simplex[n][j]);
                    }
                    double fxc = cost(xc);
                    if (fxc < f[n]) {
                        simplex[n] = xc;
                        f[n] = fxc;
                        continue;
                    }
                }
            }

            for (int i = 1; i <= n; ++i) {
                for (int j = 0; j < n; ++j) {
                    simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                }
                f[i] = cost(simplex[i]);
            }
        }

        return simplex[0];
    }
};

class MemristorFitter {
public:
    static std::vector<FitDataPoint> LoadCSV(const std::string& filepath, std::string& err) {
        std::vector<FitDataPoint> dataset;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            err = "Could not open file: " + filepath;
            return dataset;
        }

        std::string line;
        if (!std::getline(file, line)) {
            err = "Empty file: " + filepath;
            return dataset;
        }

        double last_t = 0.0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            std::vector<double> vals;
            while (std::getline(ss, item, ',')) {
                try {
                    vals.push_back(std::stod(item));
                } catch (...) {
                    // skip non-numeric items
                }
            }

            if (vals.size() >= 3) {
                FitDataPoint pt;
                pt.t = vals[0];
                pt.v = vals[1];
                pt.i = vals[2];
                dataset.push_back(pt);
                last_t = vals[0];
            } else if (vals.size() == 2) {
                FitDataPoint pt;
                pt.t = last_t;
                pt.v = vals[0];
                pt.i = vals[1];
                dataset.push_back(pt);
                last_t += 0.01;
            }
        }
        
        if (dataset.empty()) {
            err = "No data parsed. Requires Time,Voltage,Current or Voltage,Current.";
        }
        return dataset;
    }

    static void GenerateSyntheticCSV(const std::string& filepath, const MemristorParams& true_params) {
        std::ofstream file(filepath);
        file << "Time,Voltage,Current\n";

        PhysicsEngine true_phys(true_params);
        
        double dt = 0.005;
        double max_time = 2.0;
        int steps = (int)(max_time / dt);
        
        std::default_random_engine rng(12345);
        std::normal_distribution<double> noise(0.0, 0.0001);

        for (int step = 0; step < steps; ++step) {
            double t = step * dt;
            double freq = 1.0;
            double amp = 2.0;
            double v = amp * std::asin(std::sin(2.0 * M_PI * freq * t)) / (M_PI / 2.0);
            
            true_phys.update(dt, v);
            double noisy_i = true_phys.i() + noise(rng);
            file << t << "," << v << "," << noisy_i << "\n";
        }
    }

    static MemristorParams Fit(const std::vector<FitDataPoint>& dataset, const MemristorParams& base_params, double& final_mse) {
        if (dataset.empty()) return base_params;

        std::vector<double> init(4);
        init[0] = std::log10(base_params.R_on);
        init[1] = std::log10(base_params.R_off);
        init[2] = std::log10(base_params.k_on);
        init[3] = std::log10(std::abs(base_params.k_off));

        NelderMead::CostFunc cost = [&](const std::vector<double>& x) -> double {
            MemristorParams params = base_params;
            params.R_on = std::pow(10.0, x[0]);
            params.R_off = std::pow(10.0, x[1]);
            params.k_on = std::pow(10.0, x[2]);
            params.k_off = -std::pow(10.0, x[3]);

            if (params.R_on < 1.0 || params.R_on > params.R_off) return 1e12;
            if (params.R_off > 1e7) return 1e12;
            if (params.k_on < 0.1 || params.k_on > 1e6) return 1e12;
            if (std::abs(params.k_off) < 0.1 || std::abs(params.k_off) > 1e6) return 1e12;

            PhysicsEngine model(params);
            double total_se = 0.0;
            
            for (size_t i = 1; i < dataset.size(); ++i) {
                double dt = dataset[i].t - dataset[i - 1].t;
                if (dt <= 0.0) dt = 0.01;
                
                model.update(dt, dataset[i].v);
                
                double diff = model.i() - dataset[i].i;
                total_se += diff * diff;
            }
            
            return total_se / dataset.size();
        };

        std::vector<double> best_x = NelderMead::Optimize(cost, init, 1e-5, 200);

        MemristorParams fitted_params = base_params;
        fitted_params.R_on = std::pow(10.0, best_x[0]);
        fitted_params.R_off = std::pow(10.0, best_x[1]);
        fitted_params.k_on = std::pow(10.0, best_x[2]);
        fitted_params.k_off = -std::pow(10.0, best_x[3]);

        final_mse = cost(best_x);
        return fitted_params;
    }
};
