#include "Shader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

std::string Shader::readFile(const std::string& p) {
    std::ifstream f(p);
    if (!f.good()) {
        std::ifstream f2(std::string("../") + p);
        if (!f2.good()) return {};
        std::stringstream ss2; ss2 << f2.rdbuf(); return ss2.str();
    }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

bool Shader::load(const std::string& vsPath, const std::string& fsPath) {
    std::string vsSrc = readFile(vsPath);
    std::string fsSrc = readFile(fsPath);
    if (vsSrc.empty() || fsSrc.empty()) return false;
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* vss = vsSrc.c_str(); glShaderSource(vs, 1, &vss, nullptr); glCompileShader(vs);
    GLint ok = 0; glGetShaderiv(vs, GL_COMPILE_STATUS, &ok); if (!ok) return false;
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fss = fsSrc.c_str(); glShaderSource(fs, 1, &fss, nullptr); glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &ok); if (!ok) return false;
    m_program = glCreateProgram();
    glAttachShader(m_program, vs);
    glAttachShader(m_program, fs);
    glLinkProgram(m_program);
    glGetProgramiv(m_program, GL_LINK_STATUS, &ok); if (!ok) return false;
    glDeleteShader(vs); glDeleteShader(fs);
    return true;
}

void Shader::use() const { glUseProgram(m_program); }
void Shader::setMat4(const char* name, const glm::mat4& m) const { glUniformMatrix4fv(glGetUniformLocation(m_program, name), 1, GL_FALSE, glm::value_ptr(m)); }
void Shader::setVec3(const char* name, const glm::vec3& v) const { glUniform3fv(glGetUniformLocation(m_program, name), 1, &v[0]); }
void Shader::setVec4(const char* name, const glm::vec4& v) const { glUniform4fv(glGetUniformLocation(m_program, name), 1, &v[0]); }
void Shader::setFloat(const char* name, float f) const { glUniform1f(glGetUniformLocation(m_program, name), f); }

