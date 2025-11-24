#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float u_Time;
uniform float u_StateW;

float hash(float n) { return fract(sin(n) * 753.5453123); }

void main(){
    vec3 pos = aPos;
    float stability = u_StateW;
    if (pos.y < 0.45 && pos.y > -0.45) {
        float frequency = 10.0;
        float amplitude = 0.1 * (1.0 - stability);
        pos.x += sin(pos.y * frequency + 1.0) * amplitude;
        pos.z += cos(pos.y * frequency * 1.5) * amplitude;
    }
    FragPos = vec3(model * vec4(pos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}

