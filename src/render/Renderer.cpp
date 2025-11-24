#include "Renderer.h"
#include "Shader.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <vector>

void Renderer::create_fbo(int width, int height) {
    destroy_fbo();
    m_size = {width, height};
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glGenTextures(1, &m_color);
    glBindTexture(GL_TEXTURE_2D, m_color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color, 0);
    glGenRenderbuffers(1, &m_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_depth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderer::destroy_fbo() {
    if (m_depth) { glDeleteRenderbuffers(1, &m_depth); m_depth = 0; }
    if (m_color) { glDeleteTextures(1, &m_color); m_color = 0; }
    if (m_fbo) { glDeleteFramebuffers(1, &m_fbo); m_fbo = 0; }
}

void Renderer::init(int width, int height) {
    glEnable(GL_DEPTH_TEST);
    create_fbo(width, height);
    init_shaders();
    init_shapes();
}

void Renderer::resize(int width, int height) { create_fbo(width, height); }

void Renderer::begin_scene() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, m_size.x, m_size.y);
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::update_filament(double w, double power) { m_w = w; m_power = power; }

void Renderer::draw_scene() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)m_size.x / (float)m_size.y, 0.1f, 100.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(2.2f, 2.0f, 2.2f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    glm::vec3 lightPos(3.0f, 3.0f, 3.0f);
    glm::vec3 viewPos(2.2f, 2.0f, 2.2f);

    float filamentRadius = 0.05f + (float)m_w * 0.35f;
    float visualHeat = std::min((float)(m_power / 0.1), 1.0f);

    shaderFilament.use();
    shaderFilament.setMat4("projection", projection);
    shaderFilament.setMat4("view", view);
    shaderFilament.setVec3("u_Color", glm::vec3(0.8f, 0.8f, 0.9f));
    shaderFilament.setFloat("u_Power", visualHeat);
    shaderFilament.setVec3("u_LightPos", lightPos);
    shaderFilament.setVec3("u_ViewPos", viewPos);

    glm::mat4 model(1.0f);
    model = glm::scale(model, glm::vec3(filamentRadius, 1.0f, filamentRadius));
    shaderFilament.setMat4("model", model);
    glBindVertexArray(cylinderVAO);
    glDrawElements(GL_TRIANGLES, cylinderIndexCount, GL_UNSIGNED_INT, 0);

    glm::mat4 cubeModelTop(1.0f);
    cubeModelTop = glm::translate(cubeModelTop, glm::vec3(0.0f, 0.55f, 0.0f));
    cubeModelTop = glm::scale(cubeModelTop, glm::vec3(1.0f, 0.1f, 1.0f));
    shaderFilament.setMat4("model", cubeModelTop);
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glm::mat4 cubeModelBot(1.0f);
    cubeModelBot = glm::translate(cubeModelBot, glm::vec3(0.0f, -0.55f, 0.0f));
    cubeModelBot = glm::scale(cubeModelBot, glm::vec3(1.0f, 0.1f, 1.0f));
    shaderFilament.setMat4("model", cubeModelBot);
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    shaderGlass.use();
    shaderGlass.setMat4("projection", projection);
    shaderGlass.setMat4("view", view);
    shaderGlass.setVec4("u_Color", glm::vec4(0.0f, 0.8f, 0.9f, 0.2f));
    glm::mat4 oxideModel(1.0f);
    oxideModel = glm::scale(oxideModel, glm::vec3(1.2f, 1.0f, 1.2f));
    shaderGlass.setMat4("model", oxideModel);
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
}

void Renderer::end_scene() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

void Renderer::shutdown() { destroy_fbo(); }

GLuint Renderer::viewport_texture() const { return m_color; }
glm::ivec2 Renderer::viewport_size() const { return m_size; }

void Renderer::init_shaders() {
    shaderFilament.load("shaders/basic.vert", "shaders/filament.frag");
    shaderGlass.load("shaders/basic.vert", "shaders/glass.frag");
}

void Renderer::init_shapes() {
    float cubeVertices[] = {
        -0.5f,-0.5f,-0.5f, 0.0f,0.0f,-1.0f,
         0.5f,-0.5f,-0.5f, 0.0f,0.0f,-1.0f,
         0.5f, 0.5f,-0.5f, 0.0f,0.0f,-1.0f,
         0.5f, 0.5f,-0.5f, 0.0f,0.0f,-1.0f,
        -0.5f, 0.5f,-0.5f, 0.0f,0.0f,-1.0f,
        -0.5f,-0.5f,-0.5f, 0.0f,0.0f,-1.0f,

        -0.5f,-0.5f, 0.5f, 0.0f,0.0f,1.0f,
         0.5f,-0.5f, 0.5f, 0.0f,0.0f,1.0f,
         0.5f, 0.5f, 0.5f, 0.0f,0.0f,1.0f,
         0.5f, 0.5f, 0.5f, 0.0f,0.0f,1.0f,
        -0.5f, 0.5f, 0.5f, 0.0f,0.0f,1.0f,
        -0.5f,-0.5f, 0.5f, 0.0f,0.0f,1.0f,

        -0.5f, 0.5f, 0.5f,-1.0f,0.0f,0.0f,
        -0.5f, 0.5f,-0.5f,-1.0f,0.0f,0.0f,
        -0.5f,-0.5f,-0.5f,-1.0f,0.0f,0.0f,
        -0.5f,-0.5f,-0.5f,-1.0f,0.0f,0.0f,
        -0.5f,-0.5f, 0.5f,-1.0f,0.0f,0.0f,
        -0.5f, 0.5f, 0.5f,-1.0f,0.0f,0.0f,

         0.5f, 0.5f, 0.5f,1.0f,0.0f,0.0f,
         0.5f, 0.5f,-0.5f,1.0f,0.0f,0.0f,
         0.5f,-0.5f,-0.5f,1.0f,0.0f,0.0f,
         0.5f,-0.5f,-0.5f,1.0f,0.0f,0.0f,
         0.5f,-0.5f, 0.5f,1.0f,0.0f,0.0f,
         0.5f, 0.5f, 0.5f,1.0f,0.0f,0.0f,

        -0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f,
         0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f,
         0.5f,-0.5f, 0.5f,0.0f,-1.0f,0.0f,
         0.5f,-0.5f, 0.5f,0.0f,-1.0f,0.0f,
        -0.5f,-0.5f, 0.5f,0.0f,-1.0f,0.0f,
        -0.5f,-0.5f,-0.5f,0.0f,-1.0f,0.0f,

        -0.5f, 0.5f,-0.5f,0.0f,1.0f,0.0f,
         0.5f, 0.5f,-0.5f,0.0f,1.0f,0.0f,
         0.5f, 0.5f, 0.5f,0.0f,1.0f,0.0f,
         0.5f, 0.5f, 0.5f,0.0f,1.0f,0.0f,
        -0.5f, 0.5f, 0.5f,0.0f,1.0f,0.0f,
        -0.5f, 0.5f,-0.5f,0.0f,1.0f,0.0f
    };
    glGenVertexArrays(1, &cubeVAO);
    glGenBuffers(1, &cubeVBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    std::vector<float> cylVerts;
    std::vector<unsigned int> cylIndices;
    const int segments = 32;
    for (int i = 0; i <= segments; ++i) {
        float theta = (float)i / (float)segments * 2.0f * 3.1415926f;
        float x = std::cos(theta);
        float z = std::sin(theta);
        cylVerts.push_back(x * 0.5f); cylVerts.push_back(0.5f); cylVerts.push_back(z * 0.5f);
        cylVerts.push_back(x); cylVerts.push_back(0.0f); cylVerts.push_back(z);
        cylVerts.push_back(x * 0.5f); cylVerts.push_back(-0.5f); cylVerts.push_back(z * 0.5f);
        cylVerts.push_back(x); cylVerts.push_back(0.0f); cylVerts.push_back(z);
    }
    for (int i = 0; i < segments; ++i) {
        int top0 = i * 2;
        int bot0 = i * 2 + 1;
        int top1 = (i + 1) * 2;
        int bot1 = (i + 1) * 2 + 1;
        cylIndices.push_back(top0); cylIndices.push_back(bot0); cylIndices.push_back(top1);
        cylIndices.push_back(top1); cylIndices.push_back(bot0); cylIndices.push_back(bot1);
    }
    cylinderIndexCount = (int)cylIndices.size();
    glGenVertexArrays(1, &cylinderVAO);
    glGenBuffers(1, &cylinderVBO);
    glGenBuffers(1, &cylinderEBO);
    glBindVertexArray(cylinderVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cylinderVBO);
    glBufferData(GL_ARRAY_BUFFER, cylVerts.size() * sizeof(float), cylVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylinderEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cylIndices.size() * sizeof(unsigned int), cylIndices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
}
