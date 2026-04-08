mod bvh;
mod gpu;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

fn load_scene(path: &std::path::Path) -> SceneData {
    let t0 = Instant::now();
    log::info!("Loading scene: {}", path.display());

    let (document, buffers, _images) = gltf::import(path).expect("Failed to load GLTF");

    let mut total_verts = 0u32;
    let mut total_tris = 0u32;
    let mut total_meshes = 0u32;

    // Count totals
    for mesh in document.meshes() {
        total_meshes += 1;
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
            if let Some(pos) = reader.read_positions() {
                total_verts += pos.count() as u32;
            }
            if let Some(idx) = reader.read_indices() {
                total_tris += idx.into_u32().count() as u32 / 3;
            }
        }
    }

    log::info!(
        "Scene parsed in {:.1}s: {} meshes, {} verts, {} tris, {} materials",
        t0.elapsed().as_secs_f32(),
        total_meshes,
        total_verts,
        total_tris,
        document.materials().count(),
    );

    // Extract vertex data
    let t1 = Instant::now();
    let mut positions = Vec::with_capacity(total_verts as usize * 3);
    let mut normals = Vec::with_capacity(total_verts as usize * 3);
    let mut uvs = Vec::with_capacity(total_verts as usize * 2);
    let mut indices = Vec::with_capacity(total_tris as usize * 3);
    let mut tri_mat_ids = Vec::with_capacity(total_tris as usize);

    let mut vert_offset = 0u32;

    for node in document.nodes() {
        let transform = node.transform().matrix();
        let mat = glam::Mat4::from_cols_array_2d(&transform);
        let normal_mat = mat.inverse().transpose();

        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
                let mat_idx = prim.material().index().unwrap_or(0) as u32;

                let pos_iter = reader.read_positions().expect("No positions");
                let nrm_iter = reader.read_normals();
                let uv_iter = reader.read_tex_coords(0).map(|tc| tc.into_f32());

                let mut vert_count = 0u32;
                for (i, pos) in pos_iter.enumerate() {
                    let p = mat.transform_point3(glam::Vec3::from(pos));
                    positions.extend_from_slice(&[p.x, p.y, p.z]);

                    if let Some(ref nrm) = nrm_iter {
                        // Can't iterate nrm_iter here since it's consumed; use index approach below
                    }
                    vert_count += 1;
                }

                // Normals (separate pass since we can't double-iterate)
                if let Some(nrm_iter2) = reader.read_normals() {
                    for n in nrm_iter2 {
                        let nv = normal_mat.transform_vector3(glam::Vec3::from(n)).normalize();
                        normals.extend_from_slice(&[nv.x, nv.y, nv.z]);
                    }
                } else {
                    for _ in 0..vert_count {
                        normals.extend_from_slice(&[0.0, 1.0, 0.0]);
                    }
                }

                // UVs
                if let Some(uv_iter2) = reader.read_tex_coords(0).map(|tc| tc.into_f32()) {
                    for uv in uv_iter2 {
                        uvs.extend_from_slice(&uv);
                    }
                } else {
                    for _ in 0..vert_count {
                        uvs.extend_from_slice(&[0.0, 0.0]);
                    }
                }

                // Indices
                if let Some(idx_iter) = reader.read_indices() {
                    let idx_vec: Vec<u32> = idx_iter.into_u32().collect();
                    for chunk in idx_vec.chunks(3) {
                        indices.extend_from_slice(&[
                            chunk[0] + vert_offset,
                            chunk[1] + vert_offset,
                            chunk[2] + vert_offset,
                        ]);
                        tri_mat_ids.push(mat_idx);
                    }
                }

                vert_offset += vert_count;
            }
        }
    }

    log::info!(
        "Vertices extracted in {:.1}s: {} positions, {} normals, {} uvs, {} triangles",
        t1.elapsed().as_secs_f32(),
        positions.len() / 3,
        normals.len() / 3,
        uvs.len() / 2,
        indices.len() / 3,
    );

    SceneData {
        positions,
        normals,
        uvs,
        indices,
        tri_mat_ids,
        total_verts,
        total_tris,
    }
}

struct SceneData {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    indices: Vec<u32>,
    tri_mat_ids: Vec<u32>,
    total_verts: u32,
    total_tris: u32,
}

struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<gpu::GpuRenderer>,
    scene: Option<SceneData>,
    bvh: Option<bvh::BVH>,
}

impl App {
    fn new(scene: SceneData, bvh: bvh::BVH) -> Self {
        Self {
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
            renderer: None,
            scene: Some(scene),
            bvh: Some(bvh),
        }
    }

    fn init_wgpu(&mut self, window: Arc<Window>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find GPU adapter");

        let info = adapter.get_info();
        log::info!("GPU: {} ({:?})", info.name, info.backend);
        log::info!("Driver: {}", info.driver);

        // Check for features we want
        let features = adapter.features();
        log::info!("Adapter features: {:?}", features);
        log::info!("Key features:");
        log::info!("  shader-f16: {}", features.contains(wgpu::Features::SHADER_F16));
        log::info!("  subgroups: {}", features.contains(wgpu::Features::SUBGROUP));
        log::info!("  timestamp-query: {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
        log::info!("  float32-filterable: {}", features.contains(wgpu::Features::FLOAT32_FILTERABLE));

        let limits = adapter.limits();
        log::info!(
            "Limits: maxStorageBuffers={}, maxStorageTextures={}, maxComputeWG={}",
            limits.max_storage_buffers_per_shader_stage,
            limits.max_storage_textures_per_shader_stage,
            limits.max_compute_workgroup_size_x,
        );

        // Request device with features we need
        let mut required_features = wgpu::Features::empty();
        if features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if features.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ignis-native"),
                required_features,
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: limits
                        .max_storage_buffers_per_shader_stage,
                    max_storage_textures_per_shader_stage: limits
                        .max_storage_textures_per_shader_stage,
                    max_buffer_size: limits.max_buffer_size,
                    max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            },
        ))
        .expect("Failed to create device");

        log::info!("Device created: {:?}", device.features());

        // Test: load a WGSL shader from the parent directory
        let shader_path = std::path::Path::new("../pathtracer.wgsl");
        if shader_path.exists() {
            let wgsl = std::fs::read_to_string(shader_path).unwrap();
            log::info!(
                "Loaded pathtracer.wgsl: {} lines, {} bytes",
                wgsl.lines().count(),
                wgsl.len()
            );
            // Try to compile it
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pathtracer"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            log::info!("pathtracer.wgsl compiled OK");
            let _ = module; // suppress unused warning
        } else {
            log::warn!("pathtracer.wgsl not found at {:?}", shader_path);
        }

        // Configure surface
        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .unwrap();
        surface.configure(&device, &config);

        // Create GPU renderer with scene data
        if let (Some(scene), Some(bvh)) = (self.scene.as_ref(), self.bvh.as_ref()) {
            let renderer = gpu::GpuRenderer::new(&device, &queue, scene, bvh, config.width, config.height, config.format);
            self.renderer = Some(renderer);
        }

        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);
        self.window = Some(window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ignis Native (wgpu/Vulkan)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.init_wgpu(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let (Some(surface), Some(device), Some(config)) =
                    (&self.surface, &self.device, &mut self.config)
                {
                    config.width = new_size.width.max(1);
                    config.height = new_size.height.max(1);
                    surface.configure(device, config);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(device), Some(queue), Some(renderer)) =
                    (&self.surface, &self.device, &self.queue, &mut self.renderer)
                {
                    // 1. Run path tracer compute
                    renderer.render(device, queue);

                    // 2. Blit compute output to screen (tonemap + gamma in shader)
                    let frame = surface.get_current_texture().unwrap();
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("blit"),
                        });
                    renderer.blit_to_screen(&mut encoder, &view);

                    queue.submit(Some(encoder.finish()));
                    frame.present();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();

    // Load .env from parent directory (ignis-web root)
    let env_path = PathBuf::from("../.env");
    if env_path.exists() {
        dotenv::from_path(&env_path).ok();
        log::info!("Loaded .env from {}", env_path.display());
    }

    // Load scene from SCENE_PATH env var
    let scene_path = std::env::var("SCENE_PATH")
        .unwrap_or_else(|_| "../scene/scene.glb".to_string());
    log::info!("SCENE_PATH = {}", scene_path);
    let scene = load_scene(std::path::Path::new(&scene_path));
    log::info!(
        "Scene ready: {} tris, {} verts",
        scene.total_tris,
        scene.total_verts
    );

    // Build BVH
    let bvh = bvh::build_bvh(
        &scene.positions,
        &scene.indices,
        &scene.tri_mat_ids,
        |msg| log::info!("{}", msg),
    );
    log::info!(
        "BVH: {} nodes, {} sorted triangles",
        bvh.nodes.len(),
        bvh.sorted_tri_data.len()
    );

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(scene, bvh);
    event_loop.run_app(&mut app).unwrap();
}
