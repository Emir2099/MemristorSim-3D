#pragma once
#include <string>
#include <glad/glad.h>
#include <glm/glm.hpp>

class Shader {
public:
    Shader() = default;
    bool load(const std::string& vsPath, const std::string& fsPath);
    void use() const;
    void setMat4(const char* name, const glm::mat4& m) const;
    void setVec3(const char* name, const glm::vec3& v) const;
    void setVec4(const char* name, const glm::vec4& v) const;
    void setFloat(const char* name, float f) const;
    GLuint id() const { return m_program; }
private:
    static std::string readFile(const std::string& p);
    GLuint m_program = 0;
};

