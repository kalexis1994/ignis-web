/// GPU renderer: uploads scene to Vulkan, dispatches path tracer compute shader
use crate::bvh::BVH;
use crate::SceneData;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    sample_count: u32,
    frame_seed: u32,
    camera_pos: [f32; 3],
    _pad0: f32,
    camera_forward: [f32; 3],
    _pad1: f32,
    camera_right: [f32; 3],
    _pad2: f32,
    camera_up: [f32; 3],
    fov_factor: f32,
    sun_dir: [f32; 3],
    emissive_tri_count: u32,
    max_bounces: u32,
    frames_still: u32,
    aspect: f32,
    restir_enabled: u32,
    // Previous camera (for temporal)
    prev_pos: [f32; 3],
    _pad6: f32,
    prev_forward: [f32; 3],
    _pad7: f32,
    prev_right: [f32; 3],
    _pad8: f32,
    prev_up: [f32; 3],
    _pad9: f32,
    light_count: u32,
    sun_enabled: u32,
    _pad10: [u32; 2],
    // light_view_proj would be here (mat4x4) but we skip shadow map for now
    light_view_proj: [[f32; 4]; 4],
    // Punctual lights array (16 lights × 4 vec4f = 256 floats)
    lights: [[f32; 4]; 64],
}

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle
    let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VsOut;
    out.pos = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(uv.x, 1.0 - uv.y);
    return out;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let c = textureSample(src, samp, in.uv);
    // Simple Reinhard tonemap + gamma
    let lum = dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let mapped = c.rgb / (1.0 + lum);
    return vec4f(pow(mapped, vec3f(1.0/2.2)), 1.0);
}
"#;

pub struct GpuRenderer {
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    pipeline: wgpu::ComputePipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,
    bind_group0: wgpu::BindGroup,
    bind_group1: wgpu::BindGroup,
    bind_group2: wgpu::BindGroup,
    bind_group3: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    width: u32,
    height: u32,
    frame_index: u32,
}

fn create_storage_buffer(device: &wgpu::Device, data: &[u8], label: &str) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

impl GpuRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneData,
        bvh: &BVH,
        width: u32,
        height: u32,
    ) -> Self {
        let t0 = Instant::now();

        // --- Upload scene data to GPU ---

        // Pack vertices: pos.xyz + uv.x in w
        let vert_count = scene.positions.len() / 3;
        let mut gpu_positions = vec![0.0f32; vert_count * 4];
        let mut gpu_normals = vec![0.0f32; vert_count * 4];
        for v in 0..vert_count {
            gpu_positions[v * 4] = scene.positions[v * 3];
            gpu_positions[v * 4 + 1] = scene.positions[v * 3 + 1];
            gpu_positions[v * 4 + 2] = scene.positions[v * 3 + 2];
            gpu_positions[v * 4 + 3] = if v < scene.uvs.len() / 2 {
                scene.uvs[v * 2]
            } else {
                0.0
            };
            gpu_normals[v * 4] = scene.normals[v * 3];
            gpu_normals[v * 4 + 1] = scene.normals[v * 3 + 1];
            gpu_normals[v * 4 + 2] = scene.normals[v * 3 + 2];
            gpu_normals[v * 4 + 3] = if v < scene.uvs.len() / 2 {
                scene.uvs[v * 2 + 1]
            } else {
                0.0
            };
        }

        let pos_buf = create_storage_buffer(device, bytemuck::cast_slice(&gpu_positions), "vertices");
        let nrm_buf = create_storage_buffer(device, bytemuck::cast_slice(&gpu_normals), "normals");
        let tri_buf = create_storage_buffer(
            device,
            bytemuck::cast_slice(&bvh.sorted_tri_data),
            "tri_data",
        );
        let bvh_buf = create_storage_buffer(device, bytemuck::cast_slice(&bvh.nodes), "bvh_nodes");

        // Dummy materials (28 materials × 20 vec4f = 2240 floats)
        let mat_data = vec![0.0f32; 28 * 80];
        // Set base color to grey for all materials
        let mut mat_data = mat_data;
        for m in 0..28 {
            let base = m * 80;
            mat_data[base] = 0.8; // d0.x = base_r
            mat_data[base + 1] = 0.8; // d0.y = base_g
            mat_data[base + 2] = 0.8; // d0.z = base_b
            mat_data[base + 5] = 0.5; // d1.y = roughness
        }
        let mat_buf = create_storage_buffer(device, bytemuck::cast_slice(&mat_data), "materials");

        // Dummy emissive tris (empty)
        let ems_data = vec![0.0f32; 4];
        let ems_buf = create_storage_buffer(device, bytemuck::cast_slice(&ems_data), "emissive");

        // Dummy UV extra
        let uv_extra = vec![0.0f32; vert_count * 6];
        let uv_buf = create_storage_buffer(device, bytemuck::cast_slice(&uv_extra), "uv_extra");

        // Dummy env CDF
        let env_cdf = vec![0.0f32; 1024];
        let env_buf = create_storage_buffer(device, bytemuck::cast_slice(&env_cdf), "env_cdf");

        log::info!(
            "GPU buffers uploaded in {:.1}ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );

        // --- Output texture ---
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pt_output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&Default::default());

        // Dummy textures for other outputs
        let dummy_tex = |label, fmt: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let gbuf_nd_view = dummy_tex("gbuf_nd", wgpu::TextureFormat::Rgba16Float);
        let gbuf_mat_view = dummy_tex("gbuf_mat", wgpu::TextureFormat::Rgba16Float);
        let albedo_view = dummy_tex("albedo", wgpu::TextureFormat::Rgba8Unorm);
        let denoise_nd_view = dummy_tex("denoise_nd", wgpu::TextureFormat::Rgba16Float);
        let spec_view = dummy_tex("specular", wgpu::TextureFormat::Rgba16Float);

        // Dummy shadow map (depth)
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_map"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // Dummy texture array (1x1x1)
        let tex_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tex_array"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_array_view = tex_array.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let tex_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tex_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // SHaRC dummy buffers
        let sharc_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharc_dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sharc_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sharc_params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // ReSTIR dummy buffers
        let restir_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("restir_dummy"),
            size: 256,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // --- Load and compile pathtracer shader ---
        let shader_src = std::fs::read_to_string("../pathtracer.wgsl").expect("pathtracer.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pathtracer"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // --- Create bind group layouts matching pathtracer.wgsl ---
        // Group 0: uniforms + output textures + gbuffer + shadow
        let bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg0"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None },
            ],
        });

        // Group 1: scene data
        let bg1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg1"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Group 2: SHaRC + ReSTIR
        let bg2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg2"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        // Group 3: texture array + sampler
        let bg3_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg3"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt_layout"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout, &bg2_layout, &bg3_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pathtracer"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Bind groups ---
        let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg0"),
            layout: &bg0_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&output_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuf_nd_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuf_mat_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&albedo_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&denoise_nd_view) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&spec_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });

        let bind_group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg1"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: nrm_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bvh_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: mat_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: ems_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: uv_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: env_buf.as_entire_binding() },
            ],
        });

        let bind_group2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg2"),
            layout: &bg2_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sharc_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sharc_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sharc_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: restir_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: restir_dummy.as_entire_binding() },
            ],
        });

        let bind_group3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg3"),
            layout: &bg3_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&tex_array_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&tex_sampler) },
            ],
        });

        // --- Blit pipeline (fullscreen triangle, tonemap) ---
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });
        let surface_format = wgpu::TextureFormat::Bgra8Unorm; // common surface format
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        log::info!("GPU renderer initialized in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        Self {
            output_texture,
            output_view,
            pipeline,
            blit_pipeline,
            blit_bind_group,
            bind_group0,
            bind_group1,
            bind_group2,
            bind_group3,
            uniform_buf,
            width,
            height,
            frame_index: 0,
        }
    }

    pub fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Update uniforms
        let aspect = self.width as f32 / self.height as f32;
        let fov_factor = (45.0f32.to_radians() / 2.0).tan();
        let uniforms = Uniforms {
            resolution: [self.width as f32, self.height as f32],
            sample_count: 1,
            frame_seed: self.frame_index,
            camera_pos: [0.0, 5.0, 0.0],
            _pad0: 0.0,
            camera_forward: [0.0, 0.0, -1.0],
            _pad1: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            _pad2: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            fov_factor,
            sun_dir: [0.5, 0.8, 0.3],
            emissive_tri_count: 0,
            max_bounces: 3,
            frames_still: self.frame_index,
            aspect,
            restir_enabled: 0,
            prev_pos: [0.0, 5.0, 0.0],
            _pad6: 0.0,
            prev_forward: [0.0, 0.0, -1.0],
            _pad7: 0.0,
            prev_right: [1.0, 0.0, 0.0],
            _pad8: 0.0,
            prev_up: [0.0, 1.0, 0.0],
            _pad9: 0.0,
            light_count: 0,
            sun_enabled: 1,
            _pad10: [0; 2],
            light_view_proj: [[0.0; 4]; 4],
            lights: [[0.0; 4]; 64],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pt_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pathtracer"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group0, &[]);
            pass.set_bind_group(1, &self.bind_group1, &[]);
            pass.set_bind_group(2, &self.bind_group2, &[]);
            pass.set_bind_group(3, &self.bind_group3, &[]);
            pass.dispatch_workgroups(
                (self.width + 15) / 16,
                (self.height + 15) / 16,
                1,
            );
        }
        queue.submit(Some(encoder.finish()));

        self.frame_index += 1;
    }

    pub fn blit_to_screen(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &self.blit_bind_group, &[]);
        pass.draw(0..3, 0..1); // fullscreen triangle
    }
}
