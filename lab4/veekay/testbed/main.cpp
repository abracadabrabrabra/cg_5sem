#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <ctime>

#define _USE_MATH_DEFINES
#include <math.h>
#include <lodepng.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

#ifndef ASSETS_DIR
#define ASSETS_DIR "assets"
#endif

namespace {

    size_t aligned_sizeof;

    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 8;
    constexpr uint32_t max_spot_lights = 8;
    constexpr uint32_t shadow_map_size = 2048;
    constexpr uint32_t max_shadow_casting_spots = 2;

    struct Material {
        veekay::graphics::Texture* texture = nullptr;
        VkSampler sampler = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    veekay::graphics::Texture* missing_texture = nullptr;
    VkSampler missing_texture_sampler = VK_NULL_HANDLE;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct PointLight {
        veekay::vec3 position{ 0.0f, 0.0f, 0.0f };
        float intensity{ 1.0f };
        veekay::vec3 color{ 1.0f, 1.0f, 1.0f };
        float _pad0;
    };

    struct SpotLight {
        veekay::vec3 position;
        float radius;
        veekay::vec3 direction;
        float angle;
        veekay::vec3 color;
        float _pad0;
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::mat4 light_view_projection;
        veekay::vec3 camera_pos;
        float _pad0;

        veekay::vec3 ambient_color{ 0.1f, 0.1f, 0.1f };
        float _pad1;
        veekay::vec3 ambient_lights_intensity;
        float _pad2;

        veekay::vec3 sun_light_direction;
        float _pad3;
        veekay::vec3 sun_light_color;
        float _pad4;

        uint32_t point_light_count = 0;
        uint32_t spot_light_count = 0;
        uint32_t shadow_casting_spot_count = 0;
        float _pad5;


        veekay::mat4 spot_light_matrices[max_shadow_casting_spots];
    };

    struct ShadowPushConstants {
        veekay::mat4 model_matrix;
        veekay::mat4 light_view_proj;
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
        veekay::vec3 specular_color;
        float _pad1;
        float shininess;
    };

    struct Mesh {
        veekay::graphics::Buffer* vertex_buffer;
        veekay::graphics::Buffer* index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = { 1.0f, 1.0f, 1.0f };
        veekay::vec3 rotation = {};
        veekay::mat4 matrix() const;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
        Material* material = nullptr;
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        veekay::mat4 view() const;
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

    // Scene objects
    inline namespace {
        Camera camera{
            .position = {0.0f, -0.5f, -3.0f}
        };

        std::vector<Model> models;

        bool enable_ambient = true;
        bool enable_point_light = true;
        bool enable_directional_light = true;
        bool enable_spot_light = true;
        float spot_light_angle = 25.0f;

        float dir_light_intensity = 1.0f;
        float spot_light_intensity = 1.0f;
        float ambient_light_intensity = 0.3f;
        float point_light_intensity = 1.0f;

        veekay::vec3 ambient_lights_intensity{ 0.2f, 0.2f, 0.2f };

        veekay::vec3 sun_light_direction{ 3.0f, 4.0f, 0.0f };
        veekay::vec3 sun_light_color{ 1.0f, 1.0f, 1.0f };
        veekay::vec3 ambient_color{ 1.0f, 1.0f, 1.0f };
        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
        struct SpotLightAngles {
            float pitch = -M_PI / 4.0f;
            float yaw = 0.0f;
        };
        std::vector<SpotLightAngles> spot_light_angles;
    }

    // Vulkan objects
    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;
        VkShaderModule shadow_vertex_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout_ubo;
        VkDescriptorSetLayout descriptor_set_layout_tex;
        VkDescriptorSet descriptor_set_ubo;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        VkPipelineLayout shadow_pipeline_layout;
        VkPipeline shadow_pipeline;

        VkImage shadow_image;
        VkDeviceMemory shadow_image_memory;
        VkImageView shadow_image_view;
        VkSampler shadow_sampler;
        VkDescriptorSetLayout descriptor_set_layout_shadow;
        VkDescriptorSet descriptor_set_shadow;

        VkImage spot_shadow_images[max_shadow_casting_spots];
        VkDeviceMemory spot_shadow_image_memories[max_shadow_casting_spots];
        VkImageView spot_shadow_image_views[max_shadow_casting_spots];
        VkDescriptorSet descriptor_set_spot_shadows;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_buffer;
        veekay::graphics::Buffer* spot_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const {
        veekay::mat4 scale_mat = veekay::mat4::scaling(scale);
        veekay::mat4 yaw_rot = veekay::mat4::rotation({ 0.0f, 1.0f, 0.0f }, rotation.y);
        veekay::mat4 pitch_rot = veekay::mat4::rotation({ 1.0f, 0.0f, 0.0f }, rotation.x);
        veekay::mat4 roll_rot = veekay::mat4::rotation({ 0.0f, 0.0f, 1.0f }, rotation.z);
        veekay::mat4 rotation_mat = yaw_rot * pitch_rot * roll_rot;
        veekay::mat4 translation_mat = veekay::mat4::translation(position);
        return scale_mat * rotation_mat * translation_mat;
    }

    veekay::mat4 look_at_matrix(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& world_up) {
        veekay::vec3 forward = veekay::vec3::normalized(eye - target);
        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(world_up, forward));
        veekay::vec3 up = veekay::vec3::cross(forward, right);

        veekay::mat4 result;
        result[0][0] = right.x;   result[0][1] = up.x;   result[0][2] = forward.x;   result[0][3] = 0.0f;
        result[1][0] = right.y;   result[1][1] = up.y;   result[1][2] = forward.y;   result[1][3] = 0.0f;
        result[2][0] = right.z;   result[2][1] = up.z;   result[2][2] = forward.z;   result[2][3] = 0.0f;
        result[3][0] = -veekay::vec3::dot(right, eye);
        result[3][1] = -veekay::vec3::dot(up, eye);
        result[3][2] = -veekay::vec3::dot(forward, eye);
        result[3][3] = 1.0f;

        return result;
    }

    veekay::mat4 orthographic_matrix(float left, float right, float bottom, float top, float zNear, float zFar) {
        veekay::mat4 result{};
        result[0][0] = 2.0f / (right - left);
        result[1][1] = 2.0f / (bottom - top);
        result[2][2] = 1.0f / (zNear - zFar);
        result[3][3] = 1.0f;

        result[3][0] = -(right + left) / (right - left);
        result[3][1] = -(bottom + top) / (bottom - top);
        result[3][2] = zNear / (zNear - zFar);

        return result;
    }

    veekay::mat4 Camera::view() const {
        float yaw_rad = toRadians(rotation.y);
        float pitch_rad = toRadians(-rotation.x);

        veekay::vec3 front = {
            cosf(pitch_rad) * sinf(yaw_rad),
            sinf(pitch_rad),
            cosf(pitch_rad) * cosf(yaw_rad) };
        front = veekay::vec3::normalized(front);

        veekay::vec3 world_up = { 0.0f, 1.0f, 0.0f };
        veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
        veekay::vec3 up = veekay::vec3::cross(right, front);

        veekay::vec3 target = position + front;

        veekay::vec3 z = veekay::vec3::normalized(position - target);
        veekay::vec3 x = right;
        veekay::vec3 y = up;

        veekay::mat4 result = veekay::mat4::identity();
        result[0][0] = x.x;
        result[1][0] = x.y;
        result[2][0] = x.z;
        result[0][1] = y.x;
        result[1][1] = y.y;
        result[2][1] = y.z;
        result[0][2] = z.x;
        result[1][2] = z.y;
        result[2][2] = z.z;
        result[3][0] = -veekay::vec3::dot(x, position);
        result[3][1] = -veekay::vec3::dot(y, position);
        result[3][2] = -veekay::vec3::dot(z, position);

        return result;
    }

    veekay::vec3 directionFromAngles(float pitch, float yaw) {
        return veekay::vec3::normalized({
            cosf(yaw) * cosf(pitch),
            -sinf(pitch),
            sinf(yaw) * cosf(pitch)
            });
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view() * projection;
    }

    bool loadPNG(const std::string& filename, std::vector<uint8_t>& pixels, uint32_t& width, uint32_t& height) {
        unsigned error = lodepng::decode(pixels, width, height, filename);
        if (error) {
            std::cerr << "Ошибка загрузки PNG: " << filename << " : " << lodepng_error_text(error) << std::endl;
            return false;
        }
        return true;
    }

    Material* createMaterialFromFile(VkCommandBuffer cmd, const std::string& filename, VkDescriptorPool pool) {
        auto* mat = new Material();

        std::vector<uint8_t> pixels;
        uint32_t width = 0, height = 0;
        bool ok = loadPNG(filename, pixels, width, height);

        if (!ok) {
            mat->texture = missing_texture;
            mat->sampler = missing_texture_sampler;
        }
        else {
            mat->texture = new veekay::graphics::Texture(
                cmd,
                width,
                height,
                VK_FORMAT_R8G8B8A8_UNORM,
                pixels.data()
            );

            VkSamplerCreateInfo sampler_info{};
            sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sampler_info.magFilter = VK_FILTER_LINEAR;
            sampler_info.minFilter = VK_FILTER_LINEAR;
            sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            sampler_info.anisotropyEnable = VK_TRUE;
            sampler_info.maxAnisotropy = 16.0f;
            sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            sampler_info.unnormalizedCoordinates = VK_FALSE;
            sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            sampler_info.minLod = 0.0f;
            sampler_info.maxLod = 0.0f;
            sampler_info.mipLodBias = 0.0f;

            if (vkCreateSampler(veekay::app.vk_device, &sampler_info, nullptr, &mat->sampler) != VK_SUCCESS) {
                std::cerr << "Не удалось создать VkSampler для " << filename << std::endl;
                mat->sampler = missing_texture_sampler;
                mat->texture = missing_texture;
            }
        }

        VkDescriptorSetLayout layouts[] = { descriptor_set_layout_tex };
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = layouts;

        if (vkAllocateDescriptorSets(veekay::app.vk_device, &allocInfo, &mat->set) != VK_SUCCESS) {
            std::cerr << "Не удалось выделить descriptor set для материала " << filename << std::endl;
            mat->set = VK_NULL_HANDLE;
            return mat;
        }

        VkDescriptorImageInfo img{};
        img.imageView = mat->texture->view;
        img.sampler = mat->sampler;
        img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = mat->set;
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &img;

        vkUpdateDescriptorSets(veekay::app.vk_device, 1, &write, 0, nullptr);

        return mat;
    }

    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return nullptr;
        size_t size = (size_t)file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = size;
        info.pCode = buffer.data();

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }
        return result;
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        vkCreateImage(veekay::app.vk_device, &imageInfo, nullptr, &image);

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(veekay::app.vk_device, image, &memRequirements);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);

        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memoryTypeIndex = i;
                break;
            }
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        vkAllocateMemory(veekay::app.vk_device, &allocInfo, nullptr, &imageMemory);
        vkBindImageMemory(veekay::app.vk_device, image, imageMemory, 0);
    }

    void createImageView(VkImage image, VkFormat format, VkImageView& imageView, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        vkCreateImageView(veekay::app.vk_device, &viewInfo, nullptr, &imageView);
    }

    void insertImageBarrier(VkCommandBuffer cmd, VkImage image,
        VkAccessFlags srcAccess, VkAccessFlags dstAccess,
        VkImageLayout oldLayout, VkImageLayout newLayout,
        VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
        aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

        auto shaderPath = [](const char* name) { return std::string(SHADER_DIR) + "/" + name; };
        std::string vert_path = shaderPath("shader.vert.spv");
        std::string frag_path = shaderPath("shader.frag.spv");
        std::string shadow_path = shaderPath("shadow.vert.spv");

        vertex_shader_module = loadShaderModule(vert_path.c_str());
        fragment_shader_module = loadShaderModule(frag_path.c_str());
        shadow_vertex_shader_module = loadShaderModule(shadow_path.c_str());

        if (!vertex_shader_module || !fragment_shader_module || !shadow_vertex_shader_module) {
            std::cerr << "Failed to load shader modules. Expected at:\n"
                << "  " << vert_path << "\n"
                << "  " << frag_path << "\n"
                << "  " << shadow_path << "\n";
            veekay::app.running = false;
            return;
        }

        createImage(shadow_map_size, shadow_map_size, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            shadow_image, shadow_image_memory);
        createImageView(shadow_image, VK_FORMAT_D32_SFLOAT, shadow_image_view, VK_IMAGE_ASPECT_DEPTH_BIT);

        for (uint32_t i = 0; i < max_shadow_casting_spots; ++i) {
            createImage(shadow_map_size, shadow_map_size, VK_FORMAT_D32_SFLOAT,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                spot_shadow_images[i], spot_shadow_image_memories[i]);
            createImageView(spot_shadow_images[i], VK_FORMAT_D32_SFLOAT, spot_shadow_image_views[i], VK_IMAGE_ASPECT_DEPTH_BIT);
        }

        // Shadow Sampler
        {
            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            samplerInfo.mipLodBias = 0.0f;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.minLod = 0.0f;
            samplerInfo.maxLod = 1.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
            samplerInfo.compareEnable = VK_TRUE;
            samplerInfo.compareOp = VK_COMPARE_OP_LESS;

            vkCreateSampler(device, &samplerInfo, nullptr, &shadow_sampler);
        }

        VkVertexInputBindingDescription buffer_binding{};
        buffer_binding.binding = 0;
        buffer_binding.stride = sizeof(Vertex);
        buffer_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributes[] = {
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
            {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
            {.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv) },
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{};
        input_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        input_state_info.vertexBindingDescriptionCount = 1;
        input_state_info.pVertexBindingDescriptions = &buffer_binding;
        input_state_info.vertexAttributeDescriptionCount = 3;
        input_state_info.pVertexAttributeDescriptions = attributes;

        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{};
        assembly_state_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly_state_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineRasterizationStateCreateInfo raster_info{};
        raster_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster_info.polygonMode = VK_POLYGON_MODE_FILL;
        raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
        raster_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
        raster_info.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo sample_info{};
        sample_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        sample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkViewport viewport{};
        viewport.width = (float)veekay::app.window_width;
        viewport.height = (float)veekay::app.window_height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.extent = { veekay::app.window_width, veekay::app.window_height };

        VkPipelineViewportStateCreateInfo viewport_info{};
        viewport_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_info.viewportCount = 1;
        viewport_info.pViewports = &viewport;
        viewport_info.scissorCount = 1;
        viewport_info.pScissors = &scissor;

        VkPipelineDepthStencilStateCreateInfo depth_info{};
        depth_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth_info.depthTestEnable = VK_TRUE;
        depth_info.depthWriteEnable = VK_TRUE;
        depth_info.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState attachment_info{};
        attachment_info.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blend_info{};
        blend_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend_info.attachmentCount = 1;
        blend_info.pAttachments = &attachment_info;

        {
            VkDescriptorPoolSize pools[] = {
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,           1 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,   1 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,           2 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,   16 + 1 + max_shadow_casting_spots },
            };
            VkDescriptorPoolCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            info.maxSets = 1 + 16 + 2;
            info.poolSizeCount = 4;
            info.pPoolSizes = pools;
            vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
        }

        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
                {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
                {.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
            };
            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = 4;
            info.pBindings = bindings;
            vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_ubo);
        }

        {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = 1;
            info.pBindings = &binding;
            vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_tex);
        }

        {
            VkDescriptorSetLayoutBinding bindings[1 + max_shadow_casting_spots];

            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            for (uint32_t i = 0; i < max_shadow_casting_spots; ++i) {
                bindings[1 + i].binding = 1 + i;
                bindings[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[1 + i].descriptorCount = 1;
                bindings[1 + i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }

            VkDescriptorSetLayoutCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            info.bindingCount = 1 + max_shadow_casting_spots;
            info.pBindings = bindings;
            vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout_shadow);
        }

        {
            VkDescriptorSetLayout layouts[] = { descriptor_set_layout_ubo, descriptor_set_layout_tex, descriptor_set_layout_shadow };
            VkPipelineLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            layout_info.setLayoutCount = 3;
            layout_info.pSetLayouts = layouts;
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
        }

        // Основной Pipeline
        {
            VkPipelineShaderStageCreateInfo stages[2]{};

            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex_shader_module;
            stages[0].pName = "main";

            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragment_shader_module;
            stages[1].pName = "main";

            VkGraphicsPipelineCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            info.stageCount = 2;
            info.pStages = stages;
            info.pVertexInputState = &input_state_info;
            info.pInputAssemblyState = &assembly_state_info;
            info.pViewportState = &viewport_info;
            info.pRasterizationState = &raster_info;
            info.pMultisampleState = &sample_info;
            info.pDepthStencilState = &depth_info;
            info.pColorBlendState = &blend_info;
            info.layout = pipeline_layout;
            info.renderPass = veekay::app.vk_render_pass;

            vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
        }

        // Shadow Pipeline
        {
            VkPipelineShaderStageCreateInfo vertStage{};
            vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertStage.module = shadow_vertex_shader_module;
            vertStage.pName = "main";

            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(ShadowPushConstants);

            VkPipelineLayoutCreateInfo shadowLayoutInfo{};
            shadowLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            shadowLayoutInfo.pushConstantRangeCount = 1;
            shadowLayoutInfo.pPushConstantRanges = &pushConstantRange;
            shadowLayoutInfo.setLayoutCount = 0;
            vkCreatePipelineLayout(device, &shadowLayoutInfo, nullptr, &shadow_pipeline_layout);

            VkPipelineRasterizationStateCreateInfo shadowRasterInfo = raster_info;
            shadowRasterInfo.depthBiasEnable = VK_TRUE;
            shadowRasterInfo.depthBiasConstantFactor = 1.25f;
            shadowRasterInfo.depthBiasSlopeFactor = 1.75f;

            VkPipelineDepthStencilStateCreateInfo shadowDepthInfo = depth_info;
            shadowDepthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkVertexInputAttributeDescription shadowAttribute = {
                .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)
            };
            VkPipelineVertexInputStateCreateInfo shadowInputState = input_state_info;
            shadowInputState.vertexAttributeDescriptionCount = 1;
            shadowInputState.pVertexAttributeDescriptions = &shadowAttribute;

            VkViewport shadowViewport{};
            shadowViewport.width = (float)shadow_map_size;
            shadowViewport.height = (float)shadow_map_size;
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;
            VkRect2D shadowScissor{};
            shadowScissor.extent = { shadow_map_size, shadow_map_size };

            VkPipelineViewportStateCreateInfo shadowViewportState = viewport_info;
            shadowViewportState.viewportCount = 1;
            shadowViewportState.pViewports = &shadowViewport;
            shadowViewportState.scissorCount = 1;
            shadowViewportState.pScissors = &shadowScissor;

            VkPipelineRenderingCreateInfoKHR renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
            renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
            renderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
            renderingInfo.colorAttachmentCount = 0;

            VkGraphicsPipelineCreateInfo shadowInfo{};
            shadowInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            shadowInfo.stageCount = 1;
            shadowInfo.pStages = &vertStage;
            shadowInfo.pVertexInputState = &shadowInputState;
            shadowInfo.pInputAssemblyState = &assembly_state_info;
            shadowInfo.pViewportState = &shadowViewportState;
            shadowInfo.pRasterizationState = &shadowRasterInfo;
            shadowInfo.pMultisampleState = &sample_info;
            shadowInfo.pDepthStencilState = &shadowDepthInfo;
            shadowInfo.pColorBlendState = nullptr;
            shadowInfo.layout = shadow_pipeline_layout;
            shadowInfo.renderPass = VK_NULL_HANDLE;
            shadowInfo.pNext = &renderingInfo;

            vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &shadow_pipeline);
        }

        scene_uniforms_buffer = new veekay::graphics::Buffer(sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        model_uniforms_buffer = new veekay::graphics::Buffer(max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        point_lights_buffer = new veekay::graphics::Buffer(max_point_lights * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        spot_lights_buffer = new veekay::graphics::Buffer(max_spot_lights * sizeof(SpotLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Missing texture
        {
            VkSamplerCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            info.magFilter = VK_FILTER_NEAREST;
            info.minFilter = VK_FILTER_NEAREST;
            info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            vkCreateSampler(device, &info, nullptr, &missing_texture_sampler);

            uint32_t pixels[] = {
                0xff000000, 0xffff00ff,
                0xffff00ff, 0xff000000
            };
            missing_texture = new veekay::graphics::Texture(
                cmd,
                2,
                2,
                VK_FORMAT_B8G8R8A8_UNORM,
                pixels
            );
        }

        // Descriptor set для UBO/SSBO (set 0)
        {
            VkDescriptorSetLayout layouts[] = { descriptor_set_layout_ubo };
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptor_pool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = layouts;

            vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set_ubo);

            VkDescriptorBufferInfo buffer_infos[] = {
                { scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms) },
                { model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms) },
                { point_lights_buffer->buffer,   0, max_point_lights * sizeof(PointLight) },
                { spot_lights_buffer->buffer,    0, max_spot_lights * sizeof(SpotLight) },
            };

            VkWriteDescriptorSet writes[4]{};

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descriptor_set_ubo;
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[0].descriptorCount = 1;
            writes[0].pBufferInfo = &buffer_infos[0];

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descriptor_set_ubo;
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            writes[1].descriptorCount = 1;
            writes[1].pBufferInfo = &buffer_infos[1];

            writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet = descriptor_set_ubo;
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].descriptorCount = 1;
            writes[2].pBufferInfo = &buffer_infos[2];

            writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet = descriptor_set_ubo;
            writes[3].dstBinding = 3;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].descriptorCount = 1;
            writes[3].pBufferInfo = &buffer_infos[3];

            vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        }

        //  Descriptor set для всех Shadow Maps (set 2)
        {
            VkDescriptorSetLayout layouts[] = { descriptor_set_layout_shadow };
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptor_pool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = layouts;

            vkAllocateDescriptorSets(device, &allocInfo, &descriptor_set_shadow);

            VkDescriptorImageInfo imageInfos[1 + max_shadow_casting_spots];
            VkWriteDescriptorSet writes[1 + max_shadow_casting_spots];

            // Направленный свет
            imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            imageInfos[0].imageView = shadow_image_view;
            imageInfos[0].sampler = shadow_sampler;

            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].pNext = nullptr;
            writes[0].dstSet = descriptor_set_shadow;
            writes[0].dstBinding = 0;
            writes[0].dstArrayElement = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].descriptorCount = 1;
            writes[0].pImageInfo = &imageInfos[0];
            writes[0].pBufferInfo = nullptr;
            writes[0].pTexelBufferView = nullptr;

            // Прожекторы
            for (uint32_t i = 0; i < max_shadow_casting_spots; ++i) {
                imageInfos[1 + i].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                imageInfos[1 + i].imageView = spot_shadow_image_views[i];
                imageInfos[1 + i].sampler = shadow_sampler;

                writes[1 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1 + i].pNext = nullptr;
                writes[1 + i].dstSet = descriptor_set_shadow;
                writes[1 + i].dstBinding = 1 + i;
                writes[1 + i].dstArrayElement = 0;
                writes[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[1 + i].descriptorCount = 1;
                writes[1 + i].pImageInfo = &imageInfos[1 + i];
                writes[1 + i].pBufferInfo = nullptr;
                writes[1 + i].pTexelBufferView = nullptr;
            }

            vkUpdateDescriptorSets(device, 1 + max_shadow_casting_spots, writes, 0, nullptr);
        }

        // Plane mesh
        {
            std::vector<Vertex> vertices = {
                {{-5.0f, 0.0f,  5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{ 5.0f, 0.0f,  5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{ 5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            std::vector<uint32_t> idx = { 0, 1, 2, 2, 3, 0 };

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(idx.size() * sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = (uint32_t)idx.size();
        }

        // Cube mesh
        {
            std::vector<Vertex> vertices = {
                // front
                {{-0.5f, -0.5f, -0.5f}, {0,0,-1}, {0,0}},
                {{+0.5f, -0.5f, -0.5f}, {0,0,-1}, {1,0}},
                {{+0.5f, +0.5f, -0.5f}, {0,0,-1}, {1,1}},
                {{-0.5f, +0.5f, -0.5f}, {0,0,-1}, {0,1}},
                // right
                {{+0.5f, -0.5f, -0.5f}, {1,0,0}, {0,0}},
                {{+0.5f, -0.5f, +0.5f}, {1,0,0}, {1,0}},
                {{+0.5f, +0.5f, +0.5f}, {1,0,0}, {1,1}},
                {{+0.5f, +0.5f, -0.5f}, {1,0,0}, {0,1}},
                // back
                {{+0.5f, -0.5f, +0.5f}, {0,0,1}, {0,0}},
                {{-0.5f, -0.5f, +0.5f}, {0,0,1}, {1,0}},
                {{-0.5f, +0.5f, +0.5f}, {0,0,1}, {1,1}},
                {{+0.5f, +0.5f, +0.5f}, {0,0,1}, {0,1}},
                // left
                {{-0.5f, -0.5f, +0.5f}, {-1,0,0}, {0,0}},
                {{-0.5f, -0.5f, -0.5f}, {-1,0,0}, {1,0}},
                {{-0.5f, +0.5f, -0.5f}, {-1,0,0}, {1,1}},
                {{-0.5f, +0.5f, +0.5f}, {-1,0,0}, {0,1}},
                // bottom
                {{-0.5f, -0.5f, +0.5f}, {0,-1,0}, {0,0}},
                {{+0.5f, -0.5f, +0.5f}, {0,-1,0}, {1,0}},
                {{+0.5f, -0.5f, -0.5f}, {0,-1,0}, {1,1}},
                {{-0.5f, -0.5f, -0.5f}, {0,-1,0}, {0,1}},
                // top
                {{-0.5f, +0.5f, -0.5f}, {0,1,0}, {0,0}},
                {{+0.5f, +0.5f, -0.5f}, {0,1,0}, {1,0}},
                {{+0.5f, +0.5f, +0.5f}, {0,1,0}, {1,1}},
                {{-0.5f, +0.5f, +0.5f}, {0,1,0}, {0,1}},
            };

            std::vector<uint32_t> idx = {
                0,1,2,2,3,0,
                4,5,6,6,7,4,
                8,9,10,10,11,8,
                12,13,14,14,15,12,
                16,17,18,18,19,16,
                20,21,22,22,23,20
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(idx.size() * sizeof(uint32_t), idx.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = (uint32_t)idx.size();
        }

        auto assetPath = [](const char* name) { return std::string(ASSETS_DIR) + "/" + name; };
        Material* mat_ground = createMaterialFromFile(cmd, assetPath("grass.png"), descriptor_pool);
        Material* mat_red = createMaterialFromFile(cmd, assetPath("chest.png"), descriptor_pool);
        Material* mat_green = createMaterialFromFile(cmd, assetPath("minecraft.png"), descriptor_pool);
        Material* mat_blue = createMaterialFromFile(cmd, assetPath("lamp.png"), descriptor_pool);

        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .shininess = 64.0f,
            .material = mat_ground,
            });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {-2.0f, -0.5f, -1.5f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular_color = {1.0f, 1.0f, 1.0f},
            .shininess = 64.0f,
            .material = mat_red,
            });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {1.5f, -0.5f, -0.5f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular_color = {0.8f, 0.8f, 0.8f},
            .shininess = 128.0f,
            .material = mat_green,
            });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{
                .position = {0.0f, -0.5f, 1.0f},
            },
            .albedo_color = veekay::vec3{1.0f, 1.0f, 1.0f},
            .specular_color = {0.3f, 0.3f, 0.3f},
            .shininess = 16.0f,
            .material = mat_blue,
            });

        for (uint32_t i = 0; i < max_shadow_casting_spots; ++i) {
            insertImageBarrier(cmd, spot_shadow_images[i],
                0, 0,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
        }

        insertImageBarrier(cmd, shadow_image,
            0, 0,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        for (Model& m : models) {
            if (!m.material) continue;
            if (m.material->texture && m.material->texture != missing_texture) {
                delete m.material->texture;
            }
            if (m.material->sampler && m.material->sampler != missing_texture_sampler) {
                vkDestroySampler(device, m.material->sampler, nullptr);
            }
        }

        if (missing_texture_sampler) {
            vkDestroySampler(device, missing_texture_sampler, nullptr);
        }
        if (missing_texture) {
            delete missing_texture;
        }

        vkDestroyImageView(device, shadow_image_view, nullptr);
        vkFreeMemory(device, shadow_image_memory, nullptr);
        vkDestroyImage(device, shadow_image, nullptr);

        for (uint32_t i = 0; i < max_shadow_casting_spots; ++i) {
            vkDestroyImageView(device, spot_shadow_image_views[i], nullptr);
            vkFreeMemory(device, spot_shadow_image_memories[i], nullptr);
            vkDestroyImage(device, spot_shadow_images[i], nullptr);
        }

        vkDestroySampler(device, shadow_sampler, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_shadow, nullptr);
        vkDestroyPipeline(device, shadow_pipeline, nullptr);
        vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
        vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete point_lights_buffer;
        delete spot_lights_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_tex, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout_ubo, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        ImGui::Begin("Controls:");
        ImGui::Checkbox("Ambient", &enable_ambient);
        if (enable_ambient) {
            ImGui::SliderFloat("Ambient light intensity", &ambient_light_intensity, 0.0f, 5.0f);
        }
        ImGui::Checkbox("Point Light", &enable_point_light);
        if (enable_point_light) {
            ImGui::SliderFloat("Point light intensity", &point_light_intensity, 0.0f, 5.0f);
        }
        ImGui::Checkbox("Directional Light", &enable_directional_light);
        if (enable_directional_light) {
            ImGui::SliderFloat("Directional light intensity", &dir_light_intensity, 0.0f, 5.0f);
            ImGui::DragFloat3("Directional light dir", &sun_light_direction.x, 0.05f);
        }
        // Spot light management removed per request (defaults handled below)
        ImGui::End();

        if (!ImGui::IsWindowHovered()) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::right)) {
                mouse::setCaptured(true);
                auto move_delta = mouse::cursorDelta();

                camera.rotation.y += move_delta.x * 0.2f;
                camera.rotation.x += move_delta.y * 0.2f;

                if (camera.rotation.x > 89.0f)
                    camera.rotation.x = 89.0f;
                if (camera.rotation.x < -89.0f)
                    camera.rotation.x = -89.0f;
            }
            else
            {
                mouse::setCaptured(false);
            }

            float yaw_rad = toRadians(camera.rotation.y);
            float pitch_rad = toRadians(-camera.rotation.x);

            veekay::vec3 front = {
                cosf(pitch_rad) * sinf(yaw_rad),
                sinf(pitch_rad),
                cosf(pitch_rad) * cosf(yaw_rad) };
            front = veekay::vec3::normalized(front);

            veekay::vec3 world_up = { 0.0f, 1.0f, 0.0f };
            veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
            veekay::vec3 up = veekay::vec3::cross(right, front);

            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position += front * 0.1f;

            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position -= front * 0.1f;

            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right * 0.1f;

            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right * 0.1f;

            if (keyboard::isKeyDown(keyboard::Key::q))
                camera.position += up * 0.1f;

            if (keyboard::isKeyDown(keyboard::Key::e))
                camera.position -= up * 0.1f;
        }

        float yaw_rad = toRadians(camera.rotation.y);
        float pitch_rad = toRadians(-camera.rotation.x);
        veekay::vec3 spot_direction = veekay::vec3::normalized({
            cosf(pitch_rad) * sinf(yaw_rad),
            sinf(pitch_rad),
            cosf(pitch_rad) * cosf(yaw_rad) });

        // Build light lists from simple toggles/sliders
        ambient_lights_intensity = enable_ambient ? veekay::vec3{ ambient_light_intensity, ambient_light_intensity, ambient_light_intensity } : veekay::vec3{ 0.0f, 0.0f, 0.0f };

        point_lights.clear();
        if (enable_point_light) {
            point_lights.push_back(PointLight{
                .position = {0.0f, -2.0f, 0.0f},
                .intensity = point_light_intensity,
                .color = {1.0f, 1.0f, 1.0f},
                });
        }

        // No user-managed spotlights in UI; clear to avoid stale data
        spot_lights.clear();
        spot_light_angles.clear();

        float aspect_ratio = (float)veekay::app.window_width / (float)veekay::app.window_height;

        veekay::vec3 safe_sun_dir = sun_light_direction;
        if (veekay::vec3::length(safe_sun_dir) < 0.001f) {
            safe_sun_dir = { 0.0f, -1.0f, 0.0f };
        }
        veekay::vec3 light_dir = veekay::vec3::normalized(safe_sun_dir);
        veekay::vec3 light_pos = -light_dir * 20.0f;
        veekay::mat4 light_view = look_at_matrix(light_pos, { 0, 0, 0 }, { 0, 1, 0 });

        float ortho_size = 50.0f;
        float z_near = 1.0f;
        float z_far = 50.0f;
        veekay::mat4 light_proj = orthographic_matrix(-ortho_size, ortho_size, -ortho_size, ortho_size, z_near, z_far);
        veekay::mat4 light_view_projection = light_view * light_proj;

        veekay::mat4 spot_matrices[max_shadow_casting_spots];
        uint32_t shadowCastingSpotCount = std::min((uint32_t)spot_lights.size(), max_shadow_casting_spots);

        for (uint32_t i = 0; i < shadowCastingSpotCount; ++i) {
            const SpotLight& spot = spot_lights[i];
            veekay::vec3 spotTarget = spot.position - spot.direction;
            veekay::mat4 spotView = look_at_matrix(spot.position, spotTarget, { 0.0f, 1.0f, 0.0f });

            float fov = spot.angle * 2.2f;
            veekay::mat4 spotProj = veekay::mat4::projection(fov * 180.0f / M_PI, 1.0f, 0.1f, spot.radius);

            spot_matrices[i] = spotView * spotProj;
        }

        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .light_view_projection = light_view_projection,
            .camera_pos = camera.position,
            .ambient_color = ambient_color,
            .ambient_lights_intensity = ambient_lights_intensity,
            .sun_light_direction = light_dir,
            .sun_light_color = enable_directional_light ? veekay::vec3{dir_light_intensity, dir_light_intensity, dir_light_intensity} : veekay::vec3{0.0f, 0.0f, 0.0f},
            .point_light_count = (uint32_t)point_lights.size(),
            .spot_light_count = (uint32_t)spot_lights.size(),
            .shadow_casting_spot_count = shadowCastingSpotCount,
        };

        for (uint32_t i = 0; i < shadowCastingSpotCount; ++i) {
            scene_uniforms.spot_light_matrices[i] = spot_matrices[i];
        }

        *((SceneUniforms*)scene_uniforms_buffer->mapped_region) = scene_uniforms;

        if (!point_lights.empty()) {
            std::memcpy(point_lights_buffer->mapped_region, point_lights.data(),
                point_lights.size() * sizeof(PointLight));
        }
        if (!spot_lights.empty()) {
            std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(),
                spot_lights.size() * sizeof(SpotLight));
        }

        uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            ModelUniforms uniforms;
            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.specular_color = model.specular_color;
            uniforms.shininess = model.shininess;
            std::memcpy(base + i * aligned_sizeof, &uniforms, sizeof(ModelUniforms));
        }
    }



    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin_info);

        SceneUniforms* scene_data = (SceneUniforms*)scene_uniforms_buffer->mapped_region;
        veekay::mat4 light_VP = scene_data->light_view_projection;
        uint32_t shadowCastingSpotCount = scene_data->shadow_casting_spot_count;

        auto vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(
            veekay::app.vk_device, "vkCmdBeginRenderingKHR");
        auto vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(
            veekay::app.vk_device, "vkCmdEndRenderingKHR");

        {
            insertImageBarrier(cmd, shadow_image,
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

            VkRenderingAttachmentInfoKHR depthAttachment{};
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            depthAttachment.imageView = shadow_image_view;
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

            VkRenderingInfoKHR renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
            renderingInfo.renderArea = { {0, 0}, {shadow_map_size, shadow_map_size} };
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 0;
            renderingInfo.pDepthAttachment = &depthAttachment;

            if (vkCmdBeginRenderingKHR) {
                vkCmdBeginRenderingKHR(cmd, &renderingInfo);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

                VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
                VkBuffer current_index_buffer = VK_NULL_HANDLE;
                VkDeviceSize zero_offset = 0;

                for (const auto& model : models) {
                    const Mesh& mesh = model.mesh;

                    if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                        current_vertex_buffer = mesh.vertex_buffer->buffer;
                        vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
                    }
                    if (current_index_buffer != mesh.index_buffer->buffer) {
                        current_index_buffer = mesh.index_buffer->buffer;
                        vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                    }

                    ShadowPushConstants push;
                    push.model_matrix = model.transform.matrix();
                    push.light_view_proj = light_VP;

                    vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                        0, sizeof(ShadowPushConstants), &push);
                    vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
                }

                vkCmdEndRenderingKHR(cmd);
            }

            insertImageBarrier(cmd, shadow_image,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }

        for (uint32_t spotIdx = 0; spotIdx < shadowCastingSpotCount; ++spotIdx) {
            insertImageBarrier(cmd, spot_shadow_images[spotIdx],
                0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);

            VkRenderingAttachmentInfoKHR depthAttachment{};
            depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
            depthAttachment.imageView = spot_shadow_image_views[spotIdx];
            depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depthAttachment.clearValue.depthStencil = { 1.0f, 0 };

            VkRenderingInfoKHR renderingInfo{};
            renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
            renderingInfo.renderArea = { {0, 0}, {shadow_map_size, shadow_map_size} };
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 0;
            renderingInfo.pDepthAttachment = &depthAttachment;

            if (vkCmdBeginRenderingKHR) {
                vkCmdBeginRenderingKHR(cmd, &renderingInfo);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

                VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
                VkBuffer current_index_buffer = VK_NULL_HANDLE;
                VkDeviceSize zero_offset = 0;

                for (const auto& model : models) {
                    const Mesh& mesh = model.mesh;

                    if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                        current_vertex_buffer = mesh.vertex_buffer->buffer;
                        vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
                    }
                    if (current_index_buffer != mesh.index_buffer->buffer) {
                        current_index_buffer = mesh.index_buffer->buffer;
                        vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                    }

                    ShadowPushConstants push;
                    push.model_matrix = model.transform.matrix();
                    push.light_view_proj = scene_data->spot_light_matrices[spotIdx];

                    vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                        0, sizeof(ShadowPushConstants), &push);
                    vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
                }

                vkCmdEndRenderingKHR(cmd);
            }

            insertImageBarrier(cmd, spot_shadow_images[spotIdx],
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        }

        {
            VkClearValue clear_values[2];
            clear_values[0].color = { 0.1f, 0.1f, 0.1f, 1.0f };
            clear_values[1].depthStencil = { 1.0f, 0 };

            VkRenderPassBeginInfo rp_info{};
            rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp_info.renderPass = veekay::app.vk_render_pass;
            rp_info.framebuffer = framebuffer;
            rp_info.renderArea.offset = { 0, 0 };
            rp_info.renderArea.extent = { veekay::app.window_width, veekay::app.window_height };
            rp_info.clearValueCount = 2;
            rp_info.pClearValues = clear_values;

            vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            VkDeviceSize zero_offset = 0;
            VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
            VkBuffer current_index_buffer = VK_NULL_HANDLE;

            for (size_t i = 0, n = models.size(); i < n; ++i) {
                const Model& model = models[i];
                const Mesh& mesh = model.mesh;

                if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                    current_vertex_buffer = mesh.vertex_buffer->buffer;
                    vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
                }
                if (current_index_buffer != mesh.index_buffer->buffer) {
                    current_index_buffer = mesh.index_buffer->buffer;
                    vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
                }

                uint32_t offset = (uint32_t)(i * aligned_sizeof);

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                    0, 1, &descriptor_set_ubo, 1, &offset);

                if (model.material && model.material->set != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                        1, 1, &model.material->set, 0, nullptr);
                }

                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                    2, 1, &descriptor_set_shadow, 0, nullptr);

                vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
            }

            vkCmdEndRenderPass(cmd);
        }

        vkEndCommandBuffer(cmd);
    }



} // namespace

int main() {
    std::srand(time(nullptr));
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
        });
}