#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include "../physics/Memristor.h"
#include "../utils/Waveform.h"

using json = nlohmann::json;

class ConfigManager {
public:
    static void Save(const std::string& filename,
                     const MemristorParams& physParams,
                     const PulseSettings& pulseParams,
                     int waveType)
    {
        json j;
        j["physics"]["v_on"] = physParams.v_on;
        j["physics"]["v_off"] = physParams.v_off;
        j["physics"]["k_on"] = physParams.k_on;
        j["physics"]["k_off"] = physParams.k_off;
        j["physics"]["R_on"] = physParams.R_on;
        j["physics"]["R_off"] = physParams.R_off;
        j["physics"]["I_compliance"] = physParams.I_compliance;
        j["physics"]["theta_thermal"] = physParams.theta_thermal;
        j["physics"]["T_critical"] = physParams.T_critical;

        j["wave"]["type"] = waveType;
        j["wave"]["pulse_v_set"] = pulseParams.v_set;
        j["wave"]["pulse_v_reset"] = pulseParams.v_reset;
        j["wave"]["pulse_v_read"] = pulseParams.v_read;
        j["wave"]["pulse_width"] = pulseParams.pulse_width;

        std::ofstream o(filename);
        o << std::setw(4) << j << std::endl;
    }

    static bool Load(const std::string& filename,
                     MemristorParams& physParams,
                     PulseSettings& pulseParams,
                     int& waveType)
    {
        std::ifstream i(filename);
        if (!i.is_open()) return false;
        json j; i >> j;
        if (j.contains("physics")) {
            const auto& p = j["physics"];
            physParams.v_on = p.value("v_on", physParams.v_on);
            physParams.v_off = p.value("v_off", physParams.v_off);
            physParams.k_on = p.value("k_on", physParams.k_on);
            physParams.k_off = p.value("k_off", physParams.k_off);
            physParams.R_on = p.value("R_on", physParams.R_on);
            physParams.R_off = p.value("R_off", physParams.R_off);
            physParams.I_compliance = p.value("I_compliance", physParams.I_compliance);
            physParams.theta_thermal = p.value("theta_thermal", physParams.theta_thermal);
            physParams.T_critical = p.value("T_critical", physParams.T_critical);
        }
        if (j.contains("wave")) {
            const auto& w = j["wave"];
            waveType = w.value("type", waveType);
            pulseParams.v_set = w.value("pulse_v_set", pulseParams.v_set);
            pulseParams.v_reset = w.value("pulse_v_reset", pulseParams.v_reset);
            pulseParams.v_read = w.value("pulse_v_read", pulseParams.v_read);
            pulseParams.pulse_width = w.value("pulse_width", pulseParams.pulse_width);
        }
        return true;
    }
};

