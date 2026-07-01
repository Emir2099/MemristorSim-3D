#pragma once
#include <vector>
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
    
    void update(double dt) {
        // Step all devices based on row (wordline) inputs
        for (int i = 0; i < 8; ++i) {
            double v_row = m_inputs[i];
            for (int j = 0; j < 8; ++j) {
                m_devices[i][j].update(dt, v_row);
            }
        }
        
        // Perform real-time Vector-Matrix Multiplication (VMM)
        for (int j = 0; j < 8; ++j) {
            double sum_current = 0.0;
            for (int i = 0; i < 8; ++i) {
                sum_current += m_devices[i][j].i();
            }
            m_outputs[j] = sum_current;
        }
    }
    
    void program_cell(int row, int col, double w_val) {
        m_devices[row][col].set_w(w_val);
    }
    
private:
    std::vector<std::vector<PhysicsEngine>> m_devices;
    std::vector<double> m_inputs;
    std::vector<double> m_outputs;
    std::vector<double> m_ideal_outputs;
    
public:
    std::vector<std::vector<double>> m_edge_detected_output;
    std::vector<std::vector<double>> m_edge_detected_input;
    std::vector<std::vector<double>> m_kernel_weights;
};
