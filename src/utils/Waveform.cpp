#include "Waveform.h"
#include <cmath>

WaveformGenerator::WaveformGenerator() : m_waveform(Waveform::DC), m_amplitude(1.0), m_frequency(1.0) {}

double WaveformGenerator::get_voltage(double t) const {
    switch (m_waveform) {
        case Waveform::DC: return m_amplitude;
        case Waveform::Sine: return m_amplitude * std::sin(2.0 * M_PI * m_frequency * t);
        case Waveform::Triangle: {
            double x = std::fmod(t * m_frequency, 1.0);
            double tri = x < 0.5 ? (4.0 * x - 1.0) : (-4.0 * x + 3.0);
            return m_amplitude * tri;
        }
        case Waveform::Pulse: {
            double x = std::fmod(t * m_frequency, 1.0);
            return x < 0.5 ? m_amplitude : 0.0;
        }
    }
    return 0.0;
}

void WaveformGenerator::set_waveform(Waveform w) { m_waveform = w; }
void WaveformGenerator::set_amplitude(double a) { m_amplitude = a; }
void WaveformGenerator::set_frequency(double f) { m_frequency = f; }
Waveform WaveformGenerator::waveform() const { return m_waveform; }
double WaveformGenerator::amplitude() const { return m_amplitude; }
double WaveformGenerator::frequency() const { return m_frequency; }

