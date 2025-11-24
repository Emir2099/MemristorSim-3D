#pragma once

enum class Waveform { DC, Sine, Triangle, Pulse };

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
private:
    Waveform m_waveform;
    double m_amplitude;
    double m_frequency;
};

