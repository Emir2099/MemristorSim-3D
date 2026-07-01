#version 450 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec4 u_Color;
uniform vec3 u_ViewPos;

void main(){
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(u_ViewPos - FragPos);
    
    // Fresnel glow effect (higher near grazing angles/edges)
    float fresnel = pow(1.0 - max(dot(norm, viewDir), 0.0), 3.0);
    
    // Glowing edge color (vibrant neon cyan)
    vec4 edgeColor = vec4(0.0, 0.8, 1.0, 0.7);
    
    // Blend base oxide glass color and edge glow
    vec4 finalColor = mix(u_Color, edgeColor, fresnel * 0.7);
    
    // Add glowing intensity
    FragColor = vec4(finalColor.rgb, finalColor.a + fresnel * 0.3);
}
