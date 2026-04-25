//! Shared ONNX Runtime helpers.

/// Probe CUDA runtime availability and version compatibility.
///
/// The `ort` 2.0.0-rc.12 binaries can be built/downloaded for CUDA 12 or 13.
/// A mismatched CUDA major can crash inside ONNX Runtime's CUDA EP
/// initialization, so CUDA EP users should call this before registering CUDA.
pub fn probe_cuda_runtime() -> bool {
    let lib_names: &[&[u8]] = &[
        b"libcudart.so\0",
        b"libcudart.so.13\0",
        b"libcudart.so.12\0",
    ];

    let mut handle = std::ptr::null_mut();
    for name in lib_names {
        handle = unsafe { libc::dlopen(name.as_ptr() as *const libc::c_char, libc::RTLD_LAZY) };
        if !handle.is_null() {
            break;
        }
    }

    if handle.is_null() {
        tracing::error!(
            "CUDA runtime library (libcudart.so) not found. \
             Cannot initialize CUDA execution provider.\n  \
             Install the CUDA toolkit, or use a CPU backend instead."
        );
        return false;
    }

    let sym = unsafe {
        libc::dlsym(
            handle,
            b"cudaRuntimeGetVersion\0".as_ptr() as *const libc::c_char,
        )
    };

    if sym.is_null() {
        tracing::warn!("Could not find cudaRuntimeGetVersion in CUDA runtime library");
        unsafe { libc::dlclose(handle) };
        return true;
    }

    type CudaRuntimeGetVersion = unsafe extern "C" fn(*mut i32) -> i32;
    let get_version: CudaRuntimeGetVersion = unsafe { std::mem::transmute(sym) };

    let mut version: i32 = 0;
    let result = unsafe { get_version(&mut version) };
    unsafe { libc::dlclose(handle) };

    if result != 0 {
        tracing::warn!("cudaRuntimeGetVersion failed (error code {})", result);
        return true;
    }

    let major = version / 1000;
    let minor = (version % 1000) / 10;
    tracing::info!("Detected CUDA runtime version: {}.{}", major, minor);

    if !matches!(major, 12 | 13) {
        tracing::error!(
            "CUDA version mismatch: found CUDA {}.{}, but this ONNX Runtime build \
             supports CUDA 12.x or 13.x. Continuing could crash the process.\n  \
             Options:\n  \
             1. Install CUDA 12 or CUDA 13 with matching cuDNN\n  \
             2. Rebuild with ORT_CUDA_VERSION=12 or ORT_CUDA_VERSION=13 to match your system\n  \
             3. Use a CPU backend instead",
            major,
            minor,
        );
        return false;
    }

    true
}

#[cfg(feature = "onnx-common")]
pub fn maybe_apply_cuda(
    builder: ort::session::builder::SessionBuilder,
    device_id: Option<i32>,
    component: &str,
) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
    #[cfg(not(any(
        feature = "moonshine-cuda",
        feature = "sensevoice-cuda",
        feature = "paraformer-cuda",
        feature = "dolphin-cuda",
        feature = "omnilingual-cuda",
        feature = "qwen3asr-cuda",
        feature = "cohere-cuda",
        feature = "vad-onnx-cuda",
    )))]
    {
        let _ = device_id;
        let _ = component;
    }

    #[cfg(any(
        feature = "moonshine-cuda",
        feature = "sensevoice-cuda",
        feature = "paraformer-cuda",
        feature = "dolphin-cuda",
        feature = "omnilingual-cuda",
        feature = "qwen3asr-cuda",
        feature = "cohere-cuda",
        feature = "vad-onnx-cuda",
    ))]
    {
        if probe_cuda_runtime() {
            let mut cuda = ort::ep::CUDA::default()
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .with_conv_max_workspace(false);
            if let Some(device_id) = device_id {
                cuda = cuda.with_device_id(device_id);
            }
            if let Some(limit) = std::env::var("VOXTYPE_CUDA_MEM_LIMIT_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
            {
                cuda = cuda.with_memory_limit(limit * 1024 * 1024);
                tracing::info!(
                    "CUDA arena capped to {} MB via VOXTYPE_CUDA_MEM_LIMIT_MB",
                    limit
                );
            }
            tracing::info!(
                "Configuring CUDA execution provider for {}{}",
                component,
                device_id
                    .map(|id| format!(" on device {}", id))
                    .unwrap_or_default()
            );
            return Ok(builder.with_execution_providers([cuda.build()])?);
        }
        tracing::warn!(
            "CUDA not available or incompatible for {}, falling back to CPU inference",
            component
        );
    }

    #[allow(unreachable_code)]
    Ok(builder)
}
