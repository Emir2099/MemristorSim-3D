#version 450 core
out vec4 FragColor;

in vec3 NearPoint;
in vec3 FarPoint;

uniform mat4 view;
uniform mat4 projection;

float computeDepth(vec3 pos) {
    vec4 clip_space_pos = projection * view * vec4(pos.xyz, 1.0);
    return (clip_space_pos.z / clip_space_pos.w);
}

float computeLinearDepth(vec3 pos) {
    vec4 clip_space_pos = projection * view * vec4(pos.xyz, 1.0);
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w) * 2.0 - 1.0;
    float near = 0.1; float far = 100.0;
    float linearDepth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near));
    return linearDepth / far;
}

void main() {
    float t = -NearPoint.y / (FarPoint.y - NearPoint.y);
    vec3 fragPos3D = NearPoint + t * (FarPoint - NearPoint);
    gl_FragDepth = computeDepth(fragPos3D);
    if (t < 0.0) discard;

    vec2 coord = fragPos3D.xz;
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);

    vec4 color = vec4(0.3, 0.3, 0.3, 1.0 - min(line, 1.0));
    if (abs(fragPos3D.x) < 0.02) color = vec4(0.1, 0.1, 1.0, 1.0);
    if (abs(fragPos3D.z) < 0.02) color = vec4(1.0, 0.1, 0.1, 1.0);

    float linearDepth = computeLinearDepth(fragPos3D);
    float fading = max(0.0, (0.5 - linearDepth));
    FragColor = color * fading;
}

