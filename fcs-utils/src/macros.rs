//! Macros for reducing GPU shader boilerplate across the yunet codebase.
//!
//! This module provides declarative macros that eliminate repetitive patterns in WGPU code,
//! particularly for pipeline initialization, buffer readback, and uniform struct definitions.

/// Creates a read-only storage buffer bind group layout entry.
///
/// # Example
///
/// ```ignore
/// let entry = storage_buffer_entry!(0, read_only);
/// ```
#[macro_export]
macro_rules! storage_buffer_entry {
    ($binding:expr, read_only) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
    ($binding:expr, read_write) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

/// Creates a uniform buffer bind group layout entry.
///
/// # Example
///
/// ```ignore
/// let entry = uniform_buffer_entry!(2);
/// ```
#[macro_export]
macro_rules! uniform_buffer_entry {
    ($binding:expr) => {
        wgpu::BindGroupLayoutEntry {
            binding: $binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
}

/// Creates a complete GPU compute pipeline with bind group layout.
///
/// This macro encapsulates the repetitive pattern of:
/// 1. Creating a shader module from WGSL source
/// 2. Creating a bind group layout with specified entries
/// 3. Creating a pipeline layout
/// 4. Creating a compute pipeline
///
/// Returns a tuple: `(ComputePipeline, BindGroupLayout)`
///
/// # Example
///
/// ```ignore
/// use fcs_utils::{storage_buffer_entry, uniform_buffer_entry, create_gpu_pipeline};
///
/// let (pipeline, bind_group_layout) = create_gpu_pipeline!(
///     device,
///     "my_shader",
///     include_str!("shaders/my_shader.wgsl"),
///     [
///         storage_buffer_entry!(0, read_only),
///         storage_buffer_entry!(1, read_write),
///         uniform_buffer_entry!(2),
///     ]
/// );
/// ```
// Pipeline creation panics if the shader fails to compile.
// The label (e.g. "bilateral_filter.wgsl shader") is included in the wgpu panic
// message — look for that label and check the corresponding .wgsl source file.
// If the device enters a lost state, verify wgpu feature support on this GPU.
#[macro_export]
macro_rules! create_gpu_pipeline {
    ($device:expr, $label:literal, $shader_source:expr, [$($entry:expr),* $(,)?]) => {{
        let shader = $device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(concat!($label, ".wgsl shader")),
            source: wgpu::ShaderSource::Wgsl($shader_source.into()),
        });

        let bind_group_layout = $device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(concat!("yunet_", $label, "_bgl")),
            entries: &[$($entry),*],
        });

        let pipeline_layout = $device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(concat!("yunet_", $label, "_layout")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = $device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(concat!($label, ".wgsl pipeline — check wgpu feature support if this panics")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }};
}

/// Performs GPU buffer readback with async mapping and returns packed `u32` pixels.
///
/// This macro encapsulates the 11-step readback dance:
/// 1. Create buffer slice
/// 2. Create mpsc channel for async callback
/// 3. Map buffer asynchronously
/// 4. Poll device until mapping completes
/// 5. Receive mapping result
/// 6. Get mapped range
/// 7. Cast from u8 slice to `Vec<u32>`
/// 8. Drop mapped range
/// 9. Unmap buffer
/// 10. Validate output size
///
/// Returns `Result<Vec<u32>>` containing the packed RGBA pixels.
///
/// # Example
///
/// ```ignore
/// use fcs_utils::gpu_readback;
///
/// let bytes = gpu_readback!(
///     readback_buffer,
///     device,
///     expected_byte_len,
///     "gaussian blur"
/// )?;
/// ```
#[macro_export]
macro_rules! gpu_readback {
    ($readback:expr, $device:expr, $expected_len:expr, $operation:literal) => {{
        use bytemuck::cast_slice;
        use std::sync::mpsc;

        let slice = $readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        $device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|err| anyhow::anyhow!("device poll failed during {}: {err}", $operation))?;

        receiver
            .recv()
            .map_err(|_| anyhow::anyhow!("GPU {} map callback dropped", $operation))?
            .map_err(|err| anyhow::anyhow!("GPU {} map error: {err}", $operation))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|err| anyhow::anyhow!("GPU {} mapped range failed: {err}", $operation))?;
        let result_u32: Vec<u32> = cast_slice(&mapped).to_vec();
        drop(mapped);
        $readback.unmap();

        anyhow::ensure!(
            result_u32.len() == $expected_len,
            "unexpected GPU {} output size (expected {}, got {})",
            $operation,
            $expected_len,
            result_u32.len()
        );

        Ok::<Vec<u32>, anyhow::Error>(result_u32)
    }};
}

/// Defines a GPU uniform struct with manual padding specification.
///
/// All GPU uniform structs must be aligned to 16 bytes. This macro generates:
/// - `#[repr(C)]` layout for WGSL compatibility
/// - `Pod` and `Zeroable` derives from bytemuck
/// - A `__padding` field with the specified number of u32 elements
///
/// **Note**: Calculate padding as: `(16 - (field_count * 4) % 16) / 4`
/// - 3 fields (12 bytes) → 1 padding field (4 bytes)
/// - 5 fields (20 bytes) → 3 padding fields (12 bytes)
/// - 4 or 8 fields → 0 padding fields
///
/// # Example
///
/// ```ignore
/// use fcs_utils::gpu_uniforms;
///
/// gpu_uniforms!(MyUniforms, 1, {
///     width: u32,
///     height: u32,
///     radius: f32,
/// });
/// // padding = 1 because (3 fields * 4 bytes = 12 bytes, need 4 more to reach 16)
/// ```
#[macro_export]
macro_rules! gpu_uniforms {
    ($name:ident, $padding:expr, { $($field:ident: $ty:ty),+ $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct $name {
            $(pub $field: $ty,)+
            pub __padding: [u32; $padding],
        }
    };
}

#[cfg(test)]
mod tests {
    use wgpu::{BindingType, BufferBindingType, ShaderStages};

    #[test]
    fn storage_buffer_entry_read_only_sets_correct_fields() {
        let entry = storage_buffer_entry!(0, read_only);
        assert_eq!(entry.binding, 0);
        assert_eq!(entry.visibility, ShaderStages::COMPUTE);
        assert_eq!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        );
        assert!(entry.count.is_none());
    }

    #[test]
    fn storage_buffer_entry_read_write_sets_correct_fields() {
        let entry = storage_buffer_entry!(3, read_write);
        assert_eq!(entry.binding, 3);
        assert_eq!(entry.visibility, ShaderStages::COMPUTE);
        assert_eq!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        );
        assert!(entry.count.is_none());
    }

    #[test]
    fn uniform_buffer_entry_sets_correct_fields() {
        let entry = uniform_buffer_entry!(2);
        assert_eq!(entry.binding, 2);
        assert_eq!(entry.visibility, ShaderStages::COMPUTE);
        assert_eq!(
            entry.ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        );
        assert!(entry.count.is_none());
    }

    #[test]
    fn test_uniform_padding_calculation() {
        // Test 3 fields (12 bytes) -> needs 1 padding field (4 bytes) to reach 16
        gpu_uniforms!(TestUniforms3, 1, {
            a: u32,
            b: u32,
            c: u32,
        });
        assert_eq!(std::mem::size_of::<TestUniforms3>(), 16);

        // Test 5 fields (20 bytes) -> needs 3 padding fields (12 bytes) to reach 32
        gpu_uniforms!(TestUniforms5, 3, {
            a: u32,
            b: u32,
            c: u32,
            d: f32,
            e: f32,
        });
        assert_eq!(std::mem::size_of::<TestUniforms5>(), 32);

        // Test 4 fields (16 bytes) -> no padding needed
        gpu_uniforms!(TestUniforms4, 0, {
            a: u32,
            b: u32,
            c: f32,
            d: f32,
        });
        assert_eq!(std::mem::size_of::<TestUniforms4>(), 16);

        // Test 8 fields (32 bytes) -> no padding needed
        gpu_uniforms!(TestUniforms8, 0, {
            a: u32,
            b: u32,
            c: u32,
            d: u32,
            e: f32,
            f: f32,
            g: f32,
            h: f32,
        });
        assert_eq!(std::mem::size_of::<TestUniforms8>(), 32);
    }

    #[test]
    fn test_pod_zeroable_traits() {
        gpu_uniforms!(PodTestUniforms, 2, {
            x: u32,
            y: f32,
        });

        // Verify that the generated struct implements Pod and Zeroable
        let _zeroed: PodTestUniforms = bytemuck::Zeroable::zeroed();
        let bytes = bytemuck::bytes_of(&PodTestUniforms {
            x: 42,
            y: std::f32::consts::PI,
            __padding: [0; 2],
        });
        assert_eq!(bytes.len(), 16);
    }
}
