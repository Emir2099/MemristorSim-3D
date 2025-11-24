#include "Gui.h"
#include <imgui.h>
#include <implot.h>
#include <glad/glad.h>
#include <fstream>
#include <vector>
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
    switch (w) { case Waveform::DC: return 0; case Waveform::Sine: return 1; case Waveform::Triangle: return 2; case Waveform::Pulse: return 3; case Waveform::RRAM_Sequence: return 4; }
    return 0;
}

static Waveform index_to_waveform(int idx) {
    switch (idx) { case 0: return Waveform::DC; case 1: return Waveform::Sine; case 2: return Waveform::Triangle; case 3: return Waveform::Pulse; case 4: return Waveform::RRAM_Sequence; }
    return Waveform::DC;
}

void Gui::draw_controls(MemristorParams& params, WaveformGenerator& waveform, PhysicsEngine& physics) {
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
                physics.set_params(p);
                params = physics.params();
            }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::Separator();
    int wf = waveform_to_index(waveform.waveform());
    const char* items[] = {"DC","Sine","Triangle","Pulse","RRAM Sequence"};
    ImGui::Combo("Input", &wf, items, 5);
    waveform.set_waveform(index_to_waveform(wf));
    double amp = waveform.amplitude();
    double freq = waveform.frequency();
    ImGui::DragFloat("Amplitude (V)", (float*)&amp, 0.1f, -100.0f, 100.0f);
    ImGui::DragFloat("Frequency (Hz)", (float*)&freq, 0.1f, 0.01f, 1000.0f);
    waveform.set_amplitude(amp);
    waveform.set_frequency(freq);
    if (waveform.waveform() == Waveform::RRAM_Sequence) {
        ImGui::Indent();
        ImGui::TextColored(ImVec4(1,1,0,1), "Memory Cycle Config");
        PulseSettings& ps = waveform.pulse_settings();
        ImGui::DragFloat("V_SET (Write)", (float*)&ps.v_set, 0.01f, 0.0f, 5.0f);
        ImGui::DragFloat("V_RESET (Erase)", (float*)&ps.v_reset, 0.01f, -5.0f, 0.0f);
        ImGui::DragFloat("V_READ (Check)", (float*)&ps.v_read, 0.01f, 0.0f, 1.0f);
        ImGui::DragFloat("Pulse Width (s)", (float*)&ps.pulse_width, 0.01f, 0.01f, 2.0f);
        ImGui::Unindent();
    }
    ImGui::SliderFloat("Mobility (k_on)", (float*)&params.k_on, -1000.0f, -1.0f);
    ImGui::SliderFloat("Threshold (v_on)", (float*)&params.v_on, -10.0f, -0.1f);
    ImGui::SliderFloat("Compliance (A)", (float*)&params.I_compliance, 0.0001f, 0.1f, "%.6f");
    if (ImGui::Button("Reset Device")) { physics.reset(); }
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

struct TraceSample { float t; float v; float i; float r; };
struct TraceBuffer {
    int MaxSize; int Offset; ImVector<TraceSample> Data;
    TraceBuffer(int max_size = 2000) { MaxSize = max_size; Offset = 0; Data.reserve(MaxSize); }
    void Add(float t, float v, float i, float r) {
        TraceSample s{t, v, i, r};
        if (Data.size() < MaxSize) Data.push_back(s);
        else { Data[Offset] = s; Offset = (Offset + 1) % MaxSize; }
    }
};
static void SaveToCSV(const TraceBuffer& buf) {
    std::ofstream file("memristor_data.csv");
    file << "Time,Voltage,Current,Resistance\n";
    int count = buf.Data.size();
    for (int i = 0; i < count; ++i) {
        const TraceSample& s = buf.Data[i];
        file << s.t << "," << s.v << "," << s.i << "," << s.r << "\n";
    }
}

void Gui::draw_oscilloscope(double time_now, double voltage_now, const PhysicsEngine& physics) {
    ImGui::Begin("Oscilloscope");
    static TraceBuffer buf(4000);
    buf.Add((float)time_now, (float)voltage_now, (float)physics.i(), (float)physics.r());
    if (ImPlot::BeginPlot("Hysteresis Loop")) {
        ImPlot::SetupAxes("Voltage (V)", "Current (A)");
        if (buf.Data.size() > 1) {
            std::vector<float> vx(buf.Data.size());
            std::vector<float> iy(buf.Data.size());
            for (int i = 0; i < buf.Data.size(); ++i) { vx[i] = buf.Data[i].v; iy[i] = buf.Data[i].i; }
            ImPlot::PlotLine("Memristor", vx.data(), iy.data(), (int)vx.size());
        }
        ImPlot::EndPlot();
    }
    if (ImGui::Button("Export CSV")) { SaveToCSV(buf); }
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
