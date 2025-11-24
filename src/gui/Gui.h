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
    void draw_controls(MemristorParams& params, WaveformGenerator& waveform, PhysicsEngine& physics);
    void draw_viewport(unsigned int texture, glm::ivec2 size, class Camera& camera);
    void draw_oscilloscope(double time_now, double voltage_now, const PhysicsEngine& physics);
    void end_frame();
    void shutdown();
private:
    GLFWwindow* m_window;
};
