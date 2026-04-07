use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            surface: None,
            device: None,
            queue: None,
            config: None,
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
        log::info!("Features:");
        if features.contains(wgpu::Features::SHADER_F16) {
            log::info!("  - shader-f16 ✓");
        }
        if features.contains(wgpu::Features::SUBGROUP) {
            log::info!("  - subgroups ✓");
        }
        // Cooperative matrix (tensor cores) - check native Vulkan features
        log::info!("  - Checking for cooperative matrix (tensor cores)...");

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
                if let (Some(surface), Some(device), Some(queue)) =
                    (&self.surface, &self.device, &self.queue)
                {
                    let frame = surface.get_current_texture().unwrap();
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("render"),
                        });

                    // Clear to dark blue (placeholder)
                    {
                        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("clear"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.02,
                                        g: 0.02,
                                        b: 0.08,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            ..Default::default()
                        });
                    }

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
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
