#pragma once
#include <imgui.h>
#include <glm/glm.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "physics/Memristor.h"
#include "utils/Waveform.h"

class Gui {
public:
    explicit Gui(GLFWwindow* window);
    void begin_frame();
    void draw_controls(MemristorParams& params, WaveformGenerator& waveform, const PhysicsEngine& physics);
    void draw_viewport(unsigned int texture, glm::ivec2 size);
    void draw_oscilloscope(std::pair<double,double> iv);
    void end_frame();
    void shutdown();
private:
    GLFWwindow* m_window;
};
