#include "Gui.h"
#include <imgui.h>
#include <implot.h>
#include <glad/glad.h>
#include <fstream>
#include <vector>
#include "../utils/ConfigManager.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include "../render/Camera.h"
#include "physics/Optimizer.h"

Gui::Gui(GLFWwindow* window) : m_window(window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    // Style settings
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.ChildRounding = 6.0f;
    style.FrameRounding = 6.0f;
    style.PopupRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.TabRounding = 6.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.WindowPadding = ImVec2(10.0f, 10.0f);
    style.FramePadding = ImVec2(6.0f, 4.0f);
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.09f, 0.10f, 0.12f, 1.00f); // Deep Carbon
    colors[ImGuiCol_ChildBg]                = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    colors[ImGuiCol_Border]                 = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.16f, 0.17f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.22f, 0.24f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.28f, 0.30f, 0.35f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.15f, 0.17f, 0.20f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.09f, 0.10f, 0.12f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.24f, 0.26f, 0.31f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.30f, 0.32f, 0.38f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.37f, 0.40f, 0.47f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.55f, 0.45f, 0.90f, 1.00f); // Glowing Violet
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.55f, 0.45f, 0.90f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.65f, 0.55f, 0.95f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.22f, 0.25f, 0.35f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.30f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.40f, 0.45f, 0.65f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.16f, 0.20f, 0.30f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.22f, 0.27f, 0.40f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.30f, 0.35f, 0.55f, 1.00f);
    colors[ImGuiCol_Separator]              = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.30f, 0.35f, 0.50f, 1.00f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.45f, 0.65f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.24f, 0.27f, 0.38f, 1.00f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.35f, 0.40f, 0.55f, 1.00f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.45f, 0.50f, 0.70f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.14f, 0.15f, 0.18f, 1.00f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.22f, 0.24f, 0.30f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.25f, 0.38f, 1.00f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.15f, 0.18f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.00f, 0.80f, 1.00f, 1.00f); // Bright Cyan
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.80f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    
    // Style settings for ImPlot
    ImPlotStyle& plotStyle = ImPlot::GetStyle();
    plotStyle.Colors[ImPlotCol_FrameBg] = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
    plotStyle.Colors[ImPlotCol_PlotBg] = ImVec4(0.09f, 0.10f, 0.12f, 1.00f);
    plotStyle.Colors[ImPlotCol_PlotBorder] = ImVec4(0.20f, 0.22f, 0.27f, 1.00f);
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");
}

void Gui::begin_frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Create a full-screen dockspace over the main viewport
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_None);
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
    
    // Switch Mode Button
    if (m_crossbarMode) {
        if (ImGui::Button("<< Switch to Bipolar Single Device Simulator", ImVec2(-1.0f, 30.0f))) {
            m_crossbarMode = false;
        }
    } else {
        if (ImGui::Button(">> Switch to 8x8 Neuromorphic Crossbar (CIM)", ImVec2(-1.0f, 30.0f))) {
            m_crossbarMode = true;
        }
    }
    ImGui::Separator();
    
    if (m_crossbarMode) {
        // Crossbar Array Configuration
        if (ImGui::CollapsingHeader("Synaptic Weight Matrix (Conductance G)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Heatmap of Synaptic Connections (w_ij):");
            ImGui::Spacing();
            
            float cellSize = ImGui::GetContentRegionAvail().x / 8.5f;
            if (cellSize < 20.0f) cellSize = 20.0f;
            
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    double w_val = m_crossbar.w(r, c);
                    // Color goes from dark blue/grey (OFF) to bright purple/violet (ON)
                    ImVec4 cell_col = ImVec4(w_val * 0.6f + 0.1f, 0.15f, w_val * 0.8f + 0.2f, 1.0f);
                    
                    ImGui::PushStyleColor(ImGuiCol_Button, cell_col);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(cell_col.x + 0.1f, cell_col.y + 0.1f, cell_col.z + 0.1f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(cell_col.x + 0.2f, cell_col.y + 0.2f, cell_col.z + 0.2f, 1.0f));
                    
                    char cellId[16];
                    sprintf(cellId, "##cell_%d_%d", r, c);
                    if (ImGui::Button(cellId, ImVec2(cellSize, cellSize))) {
                        // Toggle synaptic state directly
                        m_crossbar.program_cell(r, c, w_val > 0.5 ? 0.0 : 1.0);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::Text("Synapse (%d, %d)", r, c);
                        ImGui::Text("State (w): %.3f", w_val);
                        ImGui::Text("Resistance: %.1f Ohms", m_crossbar.r(r, c));
                        ImGui::Text("Power: %.6f W", m_crossbar.power(r, c));
                        ImGui::Text("Temp Rise: %.1f K", m_crossbar.dT(r, c));
                        ImGui::EndTooltip();
                    }
                    
                    ImGui::PopStyleColor(3);
                    if (c < 7) ImGui::SameLine();
                }
            }
            
            ImGui::Spacing();
            ImGui::Text("Legend: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.1f, 0.15f, 0.2f, 1.0f), "OFF (0)"); ImGui::SameLine();
            ImGui::Text(" -> "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.7f, 0.15f, 1.0f, 1.0f), "ON (1)");
            
            ImGui::Spacing();
            
            // Synapse preset buttons
            ImGui::Text("Synaptic Programming Preset Kernels:");
            if (ImGui::Button("Clear Array (All OFF)", ImVec2(ImGui::GetContentRegionAvail().x * 0.48f, 24.0f))) {
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) m_crossbar.program_cell(r, c, 0.0);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Set Synapse Diagonal", ImVec2(ImGui::GetContentRegionAvail().x * 0.96f, 24.0f))) {
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) {
                        m_crossbar.program_cell(r, c, r == c ? 1.0 : 0.0);
                    }
                }
            }
            if (ImGui::Button("Randomize Synaptic Weights", ImVec2(-1.0f, 24.0f))) {
                std::default_random_engine rng(std::random_device{}());
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) m_crossbar.program_cell(r, c, dist(rng));
                }
            }
        }
        
        if (ImGui::CollapsingHeader("Analog Vector-Matrix Multiplication (CIM)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Row Input Voltages (V_wl):");
            std::vector<double> current_inputs = m_crossbar.inputs();
            
            ImGui::Columns(2, "WL_Inputs", false);
            ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.5f);
            
            for (int i = 0; i < 8; ++i) {
                char wlId[32];
                sprintf(wlId, "V_WL[%d]", i);
                float val = (float)current_inputs[i];
                if (ImGui::SliderFloat(wlId, &val, -2.0f, 2.0f, "%.2f V")) {
                    current_inputs[i] = val;
                }
                ImGui::NextColumn();
            }
            ImGui::Columns(1);
            m_crossbar.set_inputs(current_inputs);
            
            ImGui::Separator();
            ImGui::Text("Column Output Currents (I_bl):");
            const std::vector<double>& current_outputs = m_crossbar.outputs();
            
            ImGui::Columns(4, "BL_Outputs", false);
            for (int j = 0; j < 8; ++j) {
                ImGui::Text("I_BL[%d]:", j); ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.4f, 1.0f), "%.4f A", current_outputs[j]); ImGui::NextColumn();
            }
            ImGui::Columns(1);
        }
        
        if (ImGui::CollapsingHeader("Neuromorphic Edge Detection (Sobel Filter)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Memristive crossbar arrays can perform analog convolution in a single step. Here we demonstrate Sobel edge filtering on a test pattern:");
            
            static int kernel_type = 0; // 0 = Sobel X, 1 = Sobel Y
            const char* kernels[] = {"Sobel X (Vertical Edges)", "Sobel Y (Horizontal Edges)"};
            ImGui::Combo("Filter Kernel", &kernel_type, kernels, 2);
            
            if (ImGui::Button("Run Neuromorphic Convolution", ImVec2(-1.0f, 28.0f))) {
                std::vector<std::vector<double>> input_img = {
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1},
                    {0, 0, 0, 0, 1, 1, 1, 1}
                };
                
                std::vector<std::vector<double>> sobelX = {
                    {-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}
                };
                std::vector<std::vector<double>> sobelY = {
                    {-1, -2, -1},
                    { 0,  0,  0},
                    { 1,  2,  1}
                };
                
                auto selected_k = (kernel_type == 0) ? sobelX : sobelY;
                
                m_crossbar.m_edge_detected_input = input_img;
                m_crossbar.m_kernel_weights = selected_k;
                
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) {
                        m_crossbar.m_edge_detected_output[r][c] = 0.0;
                    }
                }
                
                for (int r = 1; r < 7; ++r) {
                    for (int c = 1; c < 7; ++c) {
                        double sum = 0.0;
                        for (int kr = -1; kr <= 1; ++kr) {
                            for (int kc = -1; kc <= 1; ++kc) {
                                sum += input_img[r + kr][c + kc] * selected_k[kr + 1][kc + 1];
                            }
                        }
                        m_crossbar.m_edge_detected_output[r][c] = std::abs(sum) / 4.0;
                    }
                }
                
                // Visually program the weights into the crossbar heatmap
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) {
                        if (c < 3 && r < 3) {
                            double mapped_w = (selected_k[r][c] + 2.0) / 4.0;
                            m_crossbar.program_cell(r, c, mapped_w);
                        } else {
                            m_crossbar.program_cell(r, c, 0.0);
                        }
                    }
                }
            }
            
            if (!m_crossbar.m_edge_detected_output.empty() && m_crossbar.m_edge_detected_output[1][1] != 0.0) {
                ImGui::Spacing();
                ImGui::Text("Conv Input (Contrast Step) -> Edge Output:");
                
                float pixSize = ImGui::GetContentRegionAvail().x / 18.0f;
                if (pixSize < 10.0f) pixSize = 10.0f;
                
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 8; ++c) {
                        double val = m_crossbar.m_edge_detected_input[r][c];
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(val, val, val, 1.0f));
                        ImGui::Button("##in_pix", ImVec2(pixSize, pixSize));
                        ImGui::PopStyleColor();
                        ImGui::SameLine();
                    }
                    ImGui::Dummy(ImVec2(10.0f, 0.0f)); ImGui::SameLine();
                    for (int c = 0; c < 8; ++c) {
                        double val = m_crossbar.m_edge_detected_output[r][c];
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, val * 0.8f, val, 1.0f));
                        ImGui::Button("##out_pix", ImVec2(pixSize, pixSize));
                        ImGui::PopStyleColor();
                        if (c < 7) ImGui::SameLine();
                    }
                }
            }
        }
        
        ImGui::Spacing();
        if (ImGui::Button("Reset Crossbar Array", ImVec2(-1.0f, 25.0f))) {
            m_crossbar.reset();
        }
    } else {
        // Single Device Configuration (Original controls)
        if (ImGui::CollapsingHeader("Excitation & Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
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
            if (ImGui::Combo("Input", &wf, items, 5)) {
                waveform.set_waveform(index_to_waveform(wf));
            }
            
            double amp = waveform.amplitude();
            double freq = waveform.frequency();
            if (ImGui::DragFloat("Amplitude (V)", (float*)&amp, 0.05f, -10.0f, 10.0f, "%.2f V")) {
                waveform.set_amplitude(amp);
            }
            if (ImGui::DragFloat("Frequency (Hz)", (float*)&freq, 0.1f, 0.01f, 100.0f, "%.2f Hz")) {
                waveform.set_frequency(freq);
            }
            
            if (waveform.waveform() == Waveform::RRAM_Sequence) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.1f, 1.0f), "Memory Cycle Config");
                PulseSettings& ps = waveform.pulse_settings();
                ImGui::DragFloat("V_SET (Write)", (float*)&ps.v_set, 0.05f, 0.0f, 5.0f, "%.2f V");
                ImGui::DragFloat("V_RESET (Erase)", (float*)&ps.v_reset, 0.05f, -5.0f, 0.0f, "%.2f V");
                ImGui::DragFloat("V_READ (Check)", (float*)&ps.v_read, 0.01f, 0.0f, 1.0f, "%.2f V");
                ImGui::DragFloat("Pulse Width (s)", (float*)&ps.pulse_width, 0.05f, 0.01f, 2.0f, "%.2f s");
            }
        }
        
        if (ImGui::CollapsingHeader("Physical Constants & Conduction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Mobility (k_on)", (float*)&params.k_on, 1.0f, 1000.0f, "%.1f");
            ImGui::SliderFloat("Threshold (v_on)", (float*)&params.v_on, -5.0f, -0.1f, "%.2f V");
            ImGui::SliderFloat("Compliance (A)", (float*)&params.I_compliance, 0.0001f, 0.1f, "%.5f A");
            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Conduction Model:");
            
            int model_idx = (int)params.conduction_model;
            const char* models[] = { "Sinh-Nonlinear", "Poole-Frenkel Emission", "Schottky Tunneling" };
            if (ImGui::Combo("Model Type", &model_idx, models, 3)) {
                params.conduction_model = (ConductionModel)model_idx;
            }
            
            if (params.conduction_model == ConductionModel::Sinh) {
                ImGui::SliderFloat("Sinh Factor (gamma)", (float*)&params.gamma_sinh, 0.5f, 5.0f, "%.2f");
            } else if (params.conduction_model == ConductionModel::PooleFrenkel) {
                ImGui::SliderFloat("PF Factor (beta_pf)", (float*)&params.beta_pf, 0.1f, 5.0f, "%.2f");
                ImGui::TextWrapped("Poole-Frenkel: ln(I/V) is proportional to sqrt(V). Mapped to R_off at 1V.");
            } else if (params.conduction_model == ConductionModel::Schottky) {
                ImGui::SliderFloat("Schottky Factor (beta_sc)", (float*)&params.beta_sc, 0.1f, 5.0f, "%.2f");
                ImGui::TextWrapped("Schottky Tunneling: ln(I) is proportional to sqrt(V). Mapped to R_off at 1V.");
            }
        }
        
        if (ImGui::CollapsingHeader("Real-Time Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Filament State (w): %.3f", physics.w());
            ImGui::ProgressBar((float)physics.w(), ImVec2(-1.0f, 16.0f), "");
            
            ImGui::Columns(2, "TelemetryColumns", false);
            ImGui::SetColumnWidth(0, 120.0f);
            
            ImGui::Text("Resistance (R):"); ImGui::NextColumn();
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%.1f Ohms", physics.r()); ImGui::NextColumn();
            
            ImGui::Text("Current (I):"); ImGui::NextColumn();
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.4f, 1.0f), "%.6f A", physics.i()); ImGui::NextColumn();
            
            ImGui::Text("Power (P):"); ImGui::NextColumn();
            ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.1f, 1.0f), "%.6f W", physics.power()); ImGui::NextColumn();
            
            ImGui::Text("Temp Rise (dT):"); ImGui::NextColumn();
            double temp = physics.dT();
            ImVec4 temp_color = ImVec4(0.2f, 0.9f, 0.4f, 1.0f);
            if (temp > params.T_critical) temp_color = ImVec4(1.0f, 0.1f, 0.1f, 1.0f);
            else if (temp > params.T_critical * 0.7) temp_color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);
            ImGui::TextColored(temp_color, "%.1f K / %.1f K", temp, params.T_critical); ImGui::NextColumn();
            
            ImGui::Columns(1);
            
            double current_mag = std::fabs(physics.i());
            bool is_compliant = (current_mag >= params.I_compliance * 0.99);
            bool is_thermal_decay = (temp > params.T_critical);
            
            if (is_compliant || is_thermal_decay) {
                ImGui::Spacing();
                ImGui::Text("Status Indicators:");
                if (is_compliant) {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), " -> COMPLIANCE CURRENT ACTIVE");
                }
                if (is_thermal_decay) {
                    ImGui::TextColored(ImVec4(1.0f, 0.1f, 0.1f, 1.0f), " -> THERMAL FILAMENT MELTDOWN");
                }
            }
            
            ImGui::Spacing();
            if (ImGui::Button("Reset Device", ImVec2(-1.0f, 25.0f))) {
                physics.reset();
            }
        }
    }
    
    physics.set_params(params);
    m_crossbar.set_params(params);
    ImGui::End();

    // -------------------------------------------------------------
    // PARAMETER AUTO-FITTING WINDOW
    // -------------------------------------------------------------
    ImGui::Begin("Parameter Auto-Fitting");
    
    ImGui::TextWrapped("Import experimental measurement data to extract model parameters (R_on, R_off, k_on, k_off) using a Nelder-Mead simplex optimizer.");
    ImGui::Separator();
    
    static char filepath_buf[256] = "experimental_data.csv";
    ImGui::InputText("CSV Filepath", filepath_buf, sizeof(filepath_buf));
    
    ImGui::Spacing();
    
    if (ImGui::Button("Generate Synthetic Experimental Data", ImVec2(-1.0f, 24.0f))) {
        MemristorParams true_p = params;
        true_p.R_on = 80.0;
        true_p.R_off = 12000.0;
        true_p.k_on = 250.0;
        true_p.k_off = -180.0;
        
        MemristorFitter::GenerateSyntheticCSV(filepath_buf, true_p);
        ImGui::OpenPopup("DataGenSuccess");
    }
    
    if (ImGui::BeginPopupModal("DataGenSuccess", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Synthetic data generated successfully!");
        ImGui::Text("File saved to: %s", filepath_buf);
        ImGui::Text("True values used: R_on=80.0, R_off=12000.0, k_on=250.0, k_off=-180.0");
        ImGui::Separator();
        if (ImGui::Button("OK", ImVec2(120.0f, 0.0f))) { ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    
    static std::string status_msg = "Idle";
    static std::string err_msg = "";
    static double fit_mse = 0.0;
    static MemristorParams fitted_results;
    static bool fit_completed = false;
    
    if (ImGui::Button("Run Nelder-Mead Optimization", ImVec2(-1.0f, 30.0f))) {
        status_msg = "Running...";
        err_msg = "";
        std::string err;
        auto dataset = MemristorFitter::LoadCSV(filepath_buf, err);
        if (!err.empty()) {
            err_msg = err;
            status_msg = "Failed";
        } else {
            double mse = 0.0;
            fitted_results = MemristorFitter::Fit(dataset, params, mse);
            fit_mse = mse;
            status_msg = "Completed Successfully";
            fit_completed = true;
            
            // Auto-apply the fitted parameters
            params = fitted_results;
            physics.set_params(fitted_results);
            m_crossbar.set_params(fitted_results);
        }
    }
    
    ImGui::Text("Status: "); ImGui::SameLine();
    if (status_msg == "Running...") {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "%s", status_msg.c_str());
    } else if (status_msg == "Completed Successfully") {
        ImGui::TextColored(ImVec4(0.0f, 0.9f, 0.4f, 1.0f), "%s", status_msg.c_str());
    } else if (status_msg == "Failed") {
        ImGui::TextColored(ImVec4(1.0f, 0.1f, 0.1f, 1.0f), "%s", status_msg.c_str());
    } else {
        ImGui::Text("%s", status_msg.c_str());
    }
    
    if (!err_msg.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.1f, 0.1f, 1.0f), "Error: %s", err_msg.c_str());
    }
    
    if (fit_completed) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Fitted Parameters (Auto-Applied):");
        
        ImGui::Columns(2, "FitResultsTable", true);
        ImGui::Separator();
        ImGui::Text("Parameter"); ImGui::NextColumn();
        ImGui::Text("Fitted Value"); ImGui::NextColumn();
        ImGui::Separator();
        
        ImGui::Text("R_on"); ImGui::NextColumn();
        ImGui::Text("%.2f Ohm", fitted_results.R_on); ImGui::NextColumn();
        
        ImGui::Text("R_off"); ImGui::NextColumn();
        ImGui::Text("%.2f Ohm", fitted_results.R_off); ImGui::NextColumn();
        
        ImGui::Text("k_on"); ImGui::NextColumn();
        ImGui::Text("%.2f", fitted_results.k_on); ImGui::NextColumn();
        
        ImGui::Text("k_off"); ImGui::NextColumn();
        ImGui::Text("%.2f", fitted_results.k_off); ImGui::NextColumn();
        
        ImGui::Columns(1);
        ImGui::Separator();
        ImGui::Text("Final Optimization MSE: %.6e", fit_mse);
    }
    
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
    
    if (ImPlot::BeginPlot("Hysteresis Loop", ImVec2(-1.0f, -40.0f))) {
        ImPlot::SetupAxes("Voltage (V)", "Current (A)");
        
        if (buf.Data.size() > 1) {
            std::vector<float> vx(buf.Data.size());
            std::vector<float> iy(buf.Data.size());
            for (int i = 0; i < buf.Data.size(); ++i) { 
                vx[i] = buf.Data[i].v; 
                iy[i] = buf.Data[i].i; 
            }
            
            // Set style properties on ImPlotSpec
            ImPlotSpec spec;
            spec.LineColor = ImVec4(0.0f, 0.8f, 1.0f, 1.0f); // Neon Cyan
            spec.LineWeight = 2.5f;
            
            ImPlot::PlotLine("I-V Curve", vx.data(), iy.data(), (int)vx.size(), spec);
        }
        
        ImPlot::EndPlot();
    }
    
    if (ImGui::Button("Export CSV", ImVec2(-1.0f, 25.0f))) { SaveToCSV(buf); }
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
void Gui::draw_menu(MemristorParams& params, WaveformGenerator& waveform, PhysicsEngine& physics) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save Configuration")) {
                ConfigManager::Save("experiment_config.json", physics.params(), waveform.pulse_settings(), (int)waveform.waveform());
            }
            if (ImGui::MenuItem("Load Configuration")) {
                int newType = (int)waveform.waveform();
                PulseSettings ps = waveform.pulse_settings();
                MemristorParams mp = physics.params();
                if (ConfigManager::Load("experiment_config.json", mp, ps, newType)) {
                    physics.set_params(mp);
                    params = physics.params();
                    waveform.set_waveform((Waveform)newType);
                    waveform.pulse_settings() = ps;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}
