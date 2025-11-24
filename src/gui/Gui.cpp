#include "Gui.h"
#include <imgui.h>
#include <implot.h>
#include <glad/glad.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "../render/Camera.h"

Gui::Gui(GLFWwindow* window) : m_window(window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void Gui::begin_frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

static int waveform_to_index(Waveform w) {
    switch (w) { case Waveform::DC: return 0; case Waveform::Sine: return 1; case Waveform::Triangle: return 2; case Waveform::Pulse: return 3; }
    return 0;
}

static Waveform index_to_waveform(int idx) {
    switch (idx) { case 0: return Waveform::DC; case 1: return Waveform::Sine; case 2: return Waveform::Triangle; case 3: return Waveform::Pulse; }
    return Waveform::DC;
}

void Gui::draw_controls(MemristorParams& params, WaveformGenerator& waveform, const PhysicsEngine& physics) {
    ImGui::Begin("Controls");
    static std::string current_material = "Custom";
    if (ImGui::BeginCombo("Material", current_material.c_str())) {
        auto presets = MemristorLibrary::GetPresets();
        for (auto const& kv : presets) {
            const std::string& name = kv.first;
            const MemristorParams& p = kv.second;
            bool sel = (current_material == name);
            if (ImGui::Selectable(name.c_str(), sel)) {
                current_material = name;
                const_cast<PhysicsEngine&>(physics).set_params(p);
                params = const_cast<PhysicsEngine&>(physics).params();
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::Separator();
    int wf = waveform_to_index(waveform.waveform());
    const char* items[] = {"DC","Sine","Triangle","Pulse"};
    ImGui::Combo("Input", &wf, items, 4);
    waveform.set_waveform(index_to_waveform(wf));
    double amp = waveform.amplitude();
    double freq = waveform.frequency();
    ImGui::DragFloat("Amplitude (V)", (float*)&amp, 0.1f, -100.0f, 100.0f);
    ImGui::DragFloat("Frequency (Hz)", (float*)&freq, 0.1f, 0.01f, 1000.0f);
    waveform.set_amplitude(amp);
    waveform.set_frequency(freq);
    ImGui::SliderFloat("Mobility (k_on)", (float*)&params.k_on, -1000.0f, -1.0f);
    ImGui::SliderFloat("Threshold (v_on)", (float*)&params.v_on, -10.0f, -0.1f);
    ImGui::SliderFloat("Compliance (A)", (float*)&params.I_compliance, 0.0001f, 0.1f, "%.6f");
    if (ImGui::Button("Reset Device")) { const_cast<PhysicsEngine&>(physics).reset(); }
    ImGui::Text("w=%.3f R=%.1f I=%.6f P=%.6f", physics.w(), physics.r(), physics.i(), physics.power());
    ImGui::End();
}

void Gui::draw_viewport(unsigned int texture, glm::ivec2 size, Camera& camera) {
    ImGui::Begin("Viewport");
    ImVec2 viewportSize = ImGui::GetContentRegionAvail();
    ImGui::Image((void*)(intptr_t)texture, {viewportSize.x, viewportSize.y}, {0,1},{1,0});
    if (ImGui::IsItemHovered()) {
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) camera.ProcessMouseScroll(wheel);
        if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            ImVec2 delta = ImGui::GetIO().MouseDelta;
            camera.ProcessMouseDrag(delta.x, delta.y);
        }
    }
    ImGui::End();
}

struct ScrollingBuffer {
    int MaxSize; int Offset; ImVector<ImVec2> Data;
    ScrollingBuffer(int max_size = 2000) { MaxSize = max_size; Offset = 0; Data.reserve(MaxSize); }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize) Data.push_back(ImVec2(x, y));
        else { Data[Offset] = ImVec2(x, y); Offset = (Offset + 1) % MaxSize; }
    }
};

void Gui::draw_oscilloscope(std::pair<double,double> iv) {
    ImGui::Begin("Oscilloscope");
    static ScrollingBuffer buf(2000);
    buf.AddPoint((float)iv.first, (float)iv.second);
    if (ImPlot::BeginPlot("Hysteresis Loop")) {
        ImPlot::SetupAxes("Voltage (V)", "Current (A)");
        if (buf.Data.size() > 1) {
            ImPlot::PlotLine("Memristor", &buf.Data[0].x, &buf.Data[0].y, buf.Data.size(), 0, buf.Offset);
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

void Gui::end_frame() {
    ImGui::Render();
    ImGuiIO& io = ImGui::GetIO();
    glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.05f, 0.05f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Gui::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}
