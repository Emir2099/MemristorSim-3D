#pragma once

enum class Waveform { DC, Sine, Triangle, Pulse, RRAM_Sequence };

struct PulseSettings {
    double v_set = 1.5;
    double v_reset = -1.5;
    double v_read = 0.2;
    double pulse_width = 0.5;
};

class WaveformGenerator {
public:
    WaveformGenerator();
    double get_voltage(double t) const;
    void set_waveform(Waveform w);
    void set_amplitude(double a);
    void set_frequency(double f);
    Waveform waveform() const;
    double amplitude() const;
    double frequency() const;
    PulseSettings& pulse_settings();
private:
    Waveform m_waveform;
    double m_amplitude;
    double m_frequency;
    PulseSettings m_pulse;
};
