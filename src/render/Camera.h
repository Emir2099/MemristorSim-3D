#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
public:
    glm::vec3 Target = glm::vec3(0.0f, 0.0f, 0.0f);
    float Radius = 3.0f;
    float Yaw = 45.0f;
    float Pitch = 30.0f;

    glm::mat4 GetViewMatrix() const {
        float p = glm::clamp(Pitch, -89.0f, 89.0f);
        glm::vec3 pos;
        pos.x = Radius * cos(glm::radians(Yaw)) * cos(glm::radians(p));
        pos.y = Radius * sin(glm::radians(p));
        pos.z = Radius * sin(glm::radians(Yaw)) * cos(glm::radians(p));
        return glm::lookAt(pos, Target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 GetProjectionMatrix(float aspectRatio) const {
        return glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
    }

    void ProcessMouseScroll(float yoffset) {
        Radius -= yoffset * 0.2f;
        if (Radius < 0.5f) Radius = 0.5f;
        if (Radius > 10.0f) Radius = 10.0f;
    }

    void ProcessMouseDrag(float xoffset, float yoffset) {
        Yaw -= xoffset * 0.5f;
        Pitch += yoffset * 0.5f;
    }
};
