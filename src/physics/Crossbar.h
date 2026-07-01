#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "Memristor.h"

class CrossbarArray {
public:
    CrossbarArray() {
        MemristorParams p;
        p.v_on = -0.8;
        p.v_off = 0.8;
        p.k_on = 100.0;
        p.k_off = -100.0;
        p.R_on = 100.0;
        p.R_off = 20000.0;
        p.w_init = 0.5; // Start with half-conductance state (50% formed)
        
        m_devices.resize(8, std::vector<PhysicsEngine>(8, PhysicsEngine(p)));
        m_inputs.resize(8, 0.0);
        m_outputs.resize(8, 0.0);
        m_ideal_outputs.resize(8, 0.0);
        
        m_v_row_nodes.resize(8, std::vector<double>(8, 0.0));
        m_v_col_nodes.resize(8, std::vector<double>(8, 0.0));
        
        m_edge_detected_output.resize(8, std::vector<double>(8, 0.0));
        m_edge_detected_input.resize(8, std::vector<double>(8, 0.0));
        m_kernel_weights.resize(3, std::vector<double>(3, 0.0));
        
        reset();
    }
    
    void reset() {
        for (int i = 0; i < 8; ++i) {
            m_inputs[i] = 0.0;
            m_outputs[i] = 0.0;
            m_ideal_outputs[i] = 0.0;
            for (int j = 0; j < 8; ++j) {
                m_devices[i][j].reset();
                m_v_row_nodes[i][j] = 0.0;
                m_v_col_nodes[i][j] = 0.0;
            }
        }
    }
    
    void set_inputs(const std::vector<double>& voltages) {
        if (voltages.size() == 8) {
            m_inputs = voltages;
        }
    }
    
    const std::vector<double>& inputs() const { return m_inputs; }
    const std::vector<double>& outputs() const { return m_outputs; }
    
    std::vector<double> differential_outputs() const {
        std::vector<double> diff(4, 0.0);
        for (int k = 0; k < 4; ++k) {
            diff[k] = m_outputs[2 * k] - m_outputs[2 * k + 1];
        }
        return diff;
    }
    
    double w(int row, int col) const {
        return m_devices[row][col].w();
    }
    
    double r(int row, int col) const {
        return m_devices[row][col].r();
    }
    
    double power(int row, int col) const {
        return m_devices[row][col].power();
    }
    
    double dT(int row, int col) const {
        return m_devices[row][col].dT();
    }

    PhysicsEngine& get_device(int row, int col) {
        return m_devices[row][col];
    }
    
    void set_params(const MemristorParams& p) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                m_devices[i][j].set_params(p);
            }
        }
    }

    // IR Drop parameters and getters/setters
    bool enable_ir_drop() const { return m_enable_ir_drop; }
    void set_enable_ir_drop(bool val) { m_enable_ir_drop = val; }
    
    double r_wire() const { return m_r_wire; }
    void set_r_wire(double r) { m_r_wire = r; }
    
    double v_row_node(int row, int col) const { return m_v_row_nodes[row][col]; }
    double v_col_node(int row, int col) const { return m_v_col_nodes[row][col]; }

    // DAC/ADC getters & setters
    bool enable_dac() const { return m_enable_dac; }
    void set_enable_dac(bool val) { m_enable_dac = val; }
    int dac_bits() const { return m_dac_bits; }
    void set_dac_bits(int bits) { m_dac_bits = bits; }
    double dac_v_min() const { return m_dac_v_min; }
    void set_dac_v_min(double v) { m_dac_v_min = v; }
    double dac_v_max() const { return m_dac_v_max; }
    void set_dac_v_max(double v) { m_dac_v_max = v; }

    bool enable_adc() const { return m_enable_adc; }
    void set_enable_adc(bool val) { m_enable_adc = val; }
    int adc_bits() const { return m_adc_bits; }
    void set_adc_bits(int bits) { m_adc_bits = bits; }
    double adc_i_min() const { return m_adc_i_min; }
    void set_adc_i_min(double i) { m_adc_i_min = i; }
    double adc_i_max() const { return m_adc_i_max; }
    void set_adc_i_max(double i) { m_adc_i_max = i; }

    double quantize_dac(double v) const {
        if (!m_enable_dac || m_dac_bits <= 0) return v;
        double v_min = m_dac_v_min;
        double v_max = m_dac_v_max;
        double clamped = std::max(v_min, std::min(v, v_max));
        double levels = std::pow(2.0, m_dac_bits) - 1.0;
        double step = (v_max - v_min) / levels;
        if (step <= 0.0) return clamped;
        return v_min + std::round((clamped - v_min) / step) * step;
    }

    double quantize_adc(double i) const {
        if (!m_enable_adc || m_adc_bits <= 0) return i;
        double i_min = m_adc_i_min;
        double i_max = m_adc_i_max;
        double clamped = std::max(i_min, std::min(i, i_max));
        double levels = std::pow(2.0, m_adc_bits) - 1.0;
        double step = (i_max - i_min) / levels;
        if (step <= 0.0) return clamped;
        return i_min + std::round((clamped - i_min) / step) * step;
    }
    
    void update(double dt) {
        // Quantize input voltages using DAC
        std::vector<double> active_inputs = m_inputs;
        for (int i = 0; i < 8; ++i) {
            active_inputs[i] = quantize_dac(m_inputs[i]);
        }

        // Solve row and column node voltages first based on the active DAC-quantized inputs
        solve_nodal_voltages_with_inputs(active_inputs);

        // Step all physical devices based on the actual voltage drop across them
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                double v_diff = m_v_row_nodes[i][j] - m_v_col_nodes[i][j];
                m_devices[i][j].update(dt, v_diff);
            }
        }
        
        // Compute read-out currents at the virtual ground ammeter terminals
        for (int j = 0; j < 8; ++j) {
            double raw_i = 0.0;
            if (m_enable_ir_drop) {
                // Current exiting the column j wire segment at index 7 into ground (0.0 V):
                // I_out = V_col[7][j] / r_wire
                raw_i = m_v_col_nodes[7][j] / m_r_wire;
            } else {
                // Ideal case (0-ohm lines): simply sum the nominal currents of column devices
                double sum_current = 0.0;
                for (int i = 0; i < 8; ++i) {
                    sum_current += m_devices[i][j].i();
                }
                raw_i = sum_current;
            }
            
            // Apply ADC quantization to the readout column current
            m_outputs[j] = quantize_adc(raw_i);
        }
    }
    
    void program_cell(int row, int col, double w_val) {
        m_devices[row][col].set_w(w_val);
    }
    
    std::pair<int, double> program_cell_write_verify(int row, int col, double w_val, double tolerance = 0.01, int max_pulses = 30) {
        return m_devices[row][col].program_write_verify(w_val, tolerance, max_pulses);
    }
    
private:
    void solve_nodal_voltages_with_inputs(const std::vector<double>& inputs) {
        if (!m_enable_ir_drop) {
            // Ideal crossbar: all row nodes equal input, column nodes are virtual ground
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    m_v_row_nodes[i][j] = inputs[i];
                    m_v_col_nodes[i][j] = 0.0;
                }
            }
            return;
        }

        // Iterative Modified Nodal Analysis (MNA) using Gauss-Seidel relaxation
        // Diagonally dominant grid solves extremely quickly in a few relaxation sweeps
        int max_iters = 100;
        double tolerance = 1e-6;
        double rx = m_r_wire;
        double ry = m_r_wire;

        for (int iter = 0; iter < max_iters; ++iter) {
            double max_diff = 0.0;

            // Solve KCL at Row nodes: V_row[i][j]
            for (int i = 0; i < 8; ++i) {
                double v_in = inputs[i];
                for (int j = 0; j < 8; ++j) {
                    double old_val = m_v_row_nodes[i][j];
                    double v_left = (j == 0) ? v_in : m_v_row_nodes[i][j - 1];
                    
                    double v_new = 0.0;
                    double v_diff = old_val - m_v_col_nodes[i][j];
                    double i_mem = m_devices[i][j].calculate_current(v_diff);

                    if (j == 7) {
                        // Terminal node: no right-hand segment
                        v_new = v_left - rx * i_mem;
                    } else {
                        double v_right = m_v_row_nodes[i][j + 1];
                        v_new = (v_left + v_right - rx * i_mem) / 2.0;
                    }

                    m_v_row_nodes[i][j] = v_new;
                    max_diff = std::max(max_diff, std::abs(v_new - old_val));
                }
            }

            // Solve KCL at Column nodes: V_col[i][j]
            for (int j = 0; j < 8; ++j) {
                for (int i = 0; i < 8; ++i) {
                    double old_val = m_v_col_nodes[i][j];
                    double v_new = 0.0;
                    
                    double v_diff = m_v_row_nodes[i][j] - old_val;
                    double i_mem = m_devices[i][j].calculate_current(v_diff);

                    if (i == 0) {
                        // Topmost node: no wire segment above
                        double v_down = m_v_col_nodes[1][j];
                        v_new = v_down + ry * i_mem;
                    } else if (i == 7) {
                        // Bottommost node connected to virtual ground
                        double v_up = m_v_col_nodes[6][j];
                        v_new = (v_up + ry * i_mem) / 2.0;
                    } else {
                        double v_up = m_v_col_nodes[i - 1][j];
                        double v_down = m_v_col_nodes[i + 1][j];
                        v_new = (v_up + v_down + ry * i_mem) / 2.0;
                    }

                    m_v_col_nodes[i][j] = v_new;
                    max_diff = std::max(max_diff, std::abs(v_new - old_val));
                }
            }

            if (max_diff < tolerance) {
                break;
            }
        }
    }

    std::vector<std::vector<PhysicsEngine>> m_devices;
    std::vector<double> m_inputs;
    std::vector<double> m_outputs;
    std::vector<double> m_ideal_outputs;
    
    // Nodal voltages for IR drop calculation
    std::vector<std::vector<double>> m_v_row_nodes;
    std::vector<std::vector<double>> m_v_col_nodes;
    bool m_enable_ir_drop = false;
    double m_r_wire = 1.5; // Wire segment resistance in Ohms

    // DAC & ADC Quantization properties
    bool m_enable_dac = false;
    int m_dac_bits = 8;
    double m_dac_v_min = -2.0;
    double m_dac_v_max = 2.0;
    
    bool m_enable_adc = false;
    int m_adc_bits = 8;
    double m_adc_i_min = -0.002;
    double m_adc_i_max = 0.002;

public:
    std::vector<std::vector<double>> m_edge_detected_output;
    std::vector<std::vector<double>> m_edge_detected_input;
    std::vector<std::vector<double>> m_kernel_weights;
};
