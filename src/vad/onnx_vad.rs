//! ONNX Whisper encoder-decoder VAD implementation.

use super::{VadResult, VoiceActivityDetector};
use crate::config::VadConfig;
use crate::error::VadError;
use crate::onnx_runtime;
use crate::transcribe::whisper_mel::{
    WhisperMelExtractor, WHISPER_CHUNK_FRAMES, WHISPER_CHUNK_SAMPLES,
};
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const FRAME_SECS: f32 = 0.02;
const FRAME_SAMPLES: usize = 320;

pub struct OnnxVad {
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    threshold: f32,
    min_speech_duration_ms: u32,
    mel_extractor: WhisperMelExtractor,
}

impl OnnxVad {
    pub fn new(model_path: &Path, config: &VadConfig) -> Result<Self, VadError> {
        let model_file = resolve_model_file(model_path)?;
        let threads = num_cpus::get().min(4);

        let builder = Session::builder()
            .map_err(|e| VadError::InitFailed(format!("ONNX VAD session builder failed: {}", e)))?
            .with_intra_threads(threads)
            .map_err(|e| VadError::InitFailed(format!("Failed to set ONNX VAD threads: {}", e)))?;

        let mut builder =
            onnx_runtime::maybe_apply_cuda(builder, None, "ONNX VAD").map_err(|e| {
                VadError::InitFailed(format!("Failed to configure CUDA for ONNX VAD: {}", e))
            })?;

        let session = builder.commit_from_file(&model_file).map_err(|e| {
            VadError::InitFailed(format!(
                "Failed to load ONNX VAD model from {}: {}",
                model_file.display(),
                e
            ))
        })?;

        let input_name = session
            .inputs()
            .first()
            .map(|input| input.name().to_string())
            .ok_or_else(|| VadError::InitFailed("ONNX VAD model has no inputs".to_string()))?;
        let output_name = session
            .outputs()
            .first()
            .map(|output| output.name().to_string())
            .ok_or_else(|| VadError::InitFailed("ONNX VAD model has no outputs".to_string()))?;

        tracing::info!(
            "ONNX VAD model loaded from {} (input={}, output={})",
            model_file.display(),
            input_name,
            output_name
        );

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            threshold: config.threshold.clamp(0.0, 1.0),
            min_speech_duration_ms: config.min_speech_duration_ms,
            mel_extractor: WhisperMelExtractor::new(80),
        })
    }

    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }
}

impl VoiceActivityDetector for OnnxVad {
    fn detect(&self, samples: &[f32]) -> Result<VadResult, VadError> {
        if samples.is_empty() {
            return Ok(VadResult {
                has_speech: false,
                speech_duration_secs: 0.0,
                speech_ratio: 0.0,
                rms_energy: 0.0,
            });
        }

        let mut speech_frames = 0usize;
        let mut valid_frames_total = 0usize;

        let mut session = self
            .session
            .lock()
            .map_err(|e| VadError::DetectionFailed(format!("Failed to lock ONNX VAD: {}", e)))?;

        for chunk in samples.chunks(WHISPER_CHUNK_SAMPLES) {
            let features = self.mel_extractor.extract_fixed(
                chunk,
                WHISPER_CHUNK_SAMPLES,
                WHISPER_CHUNK_FRAMES,
            );
            let input_tensor =
                Tensor::<f32>::from_array(([1usize, 80usize, WHISPER_CHUNK_FRAMES], features))
                    .map_err(|e| {
                        VadError::DetectionFailed(format!(
                            "Failed to create ONNX VAD input tensor: {}",
                            e
                        ))
                    })?;

            let inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![(
                std::borrow::Cow::Owned(self.input_name.clone()),
                input_tensor.into(),
            )];

            let outputs = session.run(inputs).map_err(|e| {
                VadError::DetectionFailed(format!("ONNX VAD inference failed: {}", e))
            })?;

            let output = outputs.get(&self.output_name).ok_or_else(|| {
                VadError::DetectionFailed(format!("ONNX VAD output '{}' missing", self.output_name))
            })?;
            let (shape, data) = output.try_extract_tensor::<f32>().map_err(|e| {
                VadError::DetectionFailed(format!("Failed to extract ONNX VAD output: {}", e))
            })?;
            let dims: &[i64] = shape;
            if dims.len() != 2 || dims[0] != 1 || dims[1] <= 0 {
                return Err(VadError::DetectionFailed(format!(
                    "Unexpected ONNX VAD output shape: {:?}",
                    dims
                )));
            }

            let output_frames = dims[1] as usize;
            let valid_frames = chunk.len().div_ceil(FRAME_SAMPLES).min(output_frames);
            valid_frames_total += valid_frames;
            speech_frames += data[..valid_frames]
                .iter()
                .filter(|&&probability| probability >= self.threshold)
                .count();
        }

        let speech_duration_secs = speech_frames as f32 * FRAME_SECS;
        let total_duration_secs = samples.len() as f32 / 16000.0;
        let speech_ratio = if valid_frames_total > 0 {
            speech_frames as f32 / valid_frames_total as f32
        } else {
            0.0
        };
        let min_speech_secs = self.min_speech_duration_ms as f32 / 1000.0;
        let has_speech = speech_duration_secs >= min_speech_secs;
        let rms_energy = Self::calculate_rms(samples);

        tracing::debug!(
            "ONNX VAD result: has_speech={}, {:.2}s speech ({:.1}% of {:.2}s), rms={:.4}",
            has_speech,
            speech_duration_secs,
            speech_ratio * 100.0,
            total_duration_secs,
            rms_energy
        );

        Ok(VadResult {
            has_speech,
            speech_duration_secs: speech_duration_secs.min(total_duration_secs),
            speech_ratio: speech_ratio.clamp(0.0, 1.0),
            rms_energy,
        })
    }
}

fn resolve_model_file(path: &Path) -> Result<PathBuf, VadError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    if path.is_dir() {
        let model = path.join("model.onnx");
        if model.exists() {
            return Ok(model);
        }
    }
    Err(VadError::ModelNotFound(format!(
        "{} (expected model.onnx file or directory containing model.onnx)",
        path.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_empty_is_zero() {
        assert_eq!(OnnxVad::calculate_rms(&[]), 0.0);
    }

    #[test]
    fn resolve_rejects_missing_path() {
        let path = PathBuf::from("/tmp/voxtype-missing-onnx-vad-model");
        assert!(resolve_model_file(&path).is_err());
    }
}
