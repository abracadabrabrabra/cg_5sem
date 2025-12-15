#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_pos_light_space;

layout (location = 4) in vec4 f_pos_spot_light_space[2];

layout (location = 0) out vec4 final_color;

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle;
    vec3 color;
    float _pad0;
};

layout (set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection;
    vec3 camera_pos;
    float _pad0;
    
    vec3 ambient_color;
    float _pad1;
    vec3 ambient_light_intensity;
    float _pad2;
    
    vec3 sun_light_direction;
    float _pad3;
    vec3 sun_light_color;
    float _pad4;

    uint point_light_count;
    uint spot_light_count;
    uint shadow_casting_spot_count;
    float _pad5;
    
    mat4 spot_light_matrices[2];
};

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad6;
    vec3 specular_color;
    float _pad7;
    float shininess;
};

layout (set = 0, binding = 2, std430) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout (set = 0, binding = 3, std430) readonly buffer SpotLightsBuffer {
    SpotLight spot_lights[];
};

layout (set = 1, binding = 0) uniform sampler2D texSampler;

layout (set = 2, binding = 0) uniform sampler2DShadow shadowMap;
layout (set = 2, binding = 1) uniform sampler2DShadow spotShadowMap0;
layout (set = 2, binding = 2) uniform sampler2DShadow spotShadowMap1;

float calculateShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir, sampler2DShadow shadowSampler) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }

    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);
    float shadow = texture(shadowSampler, vec3(projCoords.xy, projCoords.z - bias));

    return shadow;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_pos - f_position);

    vec4 texColor = texture(texSampler, f_uv);
    vec3 albedoWithTexture = albedo_color * texColor.rgb;
    
    vec3 color = ambient_light_intensity * albedoWithTexture;

    vec3 light_dir = normalize(-sun_light_direction);
    float sun_shade = max(0.0, dot(light_dir, normal));
    
    if (sun_shade > 0.0) {
        vec3 half_vector = normalize(view_dir + light_dir);
        vec3 sun_diffuse = albedoWithTexture * sun_light_color * sun_shade;
        vec3 sun_specular = specular_color * sun_light_color *
                            pow(max(0.0, dot(normal, half_vector)), shininess);
        
        float shadowFactor = calculateShadow(f_pos_light_space, normal, light_dir, shadowMap);
        color += (sun_diffuse + sun_specular) * shadowFactor;
    }

    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];
        vec3 ldir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);
        float light_falloff = light.intensity / (distance * distance + 0.0001);
        float light_shade = max(0.0, dot(normal, ldir));
        
        vec3 light_diffuse = albedoWithTexture * light.color * light_shade;
        vec3 half_vec = normalize(ldir + view_dir);
        float spec_angle = max(0.0, dot(normal, half_vec));
        vec3 light_spec = specular_color * light.color * pow(spec_angle, shininess);

        color += light_falloff * (light_diffuse + light_spec);
    }

    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        vec3 ldir = normalize(light.position - f_position);
        float distance = length(light.position - f_position);

        if (distance > light.radius) continue;

        vec3 spot_dir = normalize(light.direction);
        float theta = dot(-ldir, spot_dir);
        float cutoff_cos = cos(light.angle);

        if (theta > cutoff_cos) {
            float distance_attenuation = clamp(1.0 - (distance / light.radius), 0.0, 1.0);
            float outer_cutoff = cos(light.angle);
            float inner_cutoff = cos(light.angle * 0.8);
            float epsilon = inner_cutoff - outer_cutoff;
            float spot_intensity = clamp((theta - outer_cutoff) / epsilon, 0.0, 1.0);
            float total_attenuation = distance_attenuation * spot_intensity;

            float light_shade = max(0.0, dot(normal, ldir));
            vec3 light_diffuse = albedoWithTexture * light.color * light_shade;

            vec3 half_vec = normalize(ldir + view_dir);
            float spec_angle = max(0.0, dot(normal, half_vec));
            vec3 light_spec = specular_color * light.color * pow(spec_angle, shininess);

            float spotShadow = 1.0;
            if (i < shadow_casting_spot_count) {
                if (i == 0) {
                    spotShadow = calculateShadow(f_pos_spot_light_space[0], normal, ldir, spotShadowMap0);
                } else if (i == 1) {
                    spotShadow = calculateShadow(f_pos_spot_light_space[1], normal, ldir, spotShadowMap1);
                }
            }

            color += total_attenuation * (light_diffuse + light_spec) * spotShadow;
        }
    }

    final_color = vec4(color, texColor.a);
}