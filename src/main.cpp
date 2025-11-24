#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "gui/Gui.h"
#include "render/Renderer.h"
#include "render/Camera.h"
#include "physics/Memristor.h"
#include "utils/Waveform.h"

static void glfw_error_callback(int error, const char* description) {
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow* window = glfwCreateWindow(1280, 800, "MemristorSim 3D", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { glfwDestroyWindow(window); glfwTerminate(); return -1; }

    Gui gui(window);
    Renderer renderer;
    renderer.init(1280, 800);
    Camera camera;

    MemristorParams params;
    PhysicsEngine physics(params);
    WaveformGenerator waveform;

    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        renderer.resize(display_w, display_h);

        double now = glfwGetTime();
        double dt = now - lastTime;
        lastTime = now;

        double voltage = waveform.get_voltage(now);
        physics.update(dt, voltage);

        renderer.begin_scene();
        renderer.update_filament(physics.w(), physics.power());
        renderer.draw_scene(camera);
        renderer.end_scene();

        gui.begin_frame();
        gui.draw_controls(params, waveform, physics);
        gui.draw_viewport(renderer.viewport_texture(), renderer.viewport_size(), camera);
        gui.draw_oscilloscope(now, voltage, physics);
        gui.end_frame();

        glfwSwapBuffers(window);
    }

    renderer.shutdown();
    gui.shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
