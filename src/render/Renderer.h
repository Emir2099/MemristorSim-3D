#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Shader.h"

class Renderer {
public:
    void init(int width, int height);
    void resize(int width, int height);
    void begin_scene();
    void update_filament(double w, double power);
    void draw_scene();
    void end_scene();
    void shutdown();
    GLuint viewport_texture() const;
    glm::ivec2 viewport_size() const;
private:
    void create_fbo(int width, int height);
    void destroy_fbo();
    void init_shapes();
    void init_shaders();
    GLuint m_fbo = 0;
    GLuint m_color = 0;
    GLuint m_depth = 0;
    glm::ivec2 m_size{0,0};
    double m_w = 0.0;
    double m_power = 0.0;
    GLuint cubeVAO = 0, cubeVBO = 0;
    GLuint cylinderVAO = 0, cylinderVBO = 0, cylinderEBO = 0;
    int cylinderIndexCount = 0;
    Shader shaderFilament;
    Shader shaderGlass;
};
