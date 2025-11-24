#version 450 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 u_Color;
uniform float u_Power;
uniform vec3 u_LightPos;
uniform vec3 u_ViewPos;
uniform float u_Conductance;

void main(){
    vec3 ambient = 0.3 * u_Color;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(u_LightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * u_Color;
    vec3 viewDir = normalize(u_ViewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = vec3(0.5) * spec;
    vec3 coldColor = ambient + diffuse + specular;
    vec3 hotColor = vec3(1.0, 0.6, 0.1);
    vec3 finalColor = mix(coldColor, hotColor, clamp(u_Power * 2.0, 0.0, 1.0));
    float gap_threshold = 0.05;
    if (u_Conductance < gap_threshold) {
        float gap_size = (gap_threshold - u_Conductance) * 2.0;
        if (abs(FragPos.y) < gap_size) { discard; }
    }
    FragColor = vec4(finalColor, 1.0);
}
