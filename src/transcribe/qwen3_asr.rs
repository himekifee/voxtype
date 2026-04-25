//! Qwen3-ASR ONNX transcription.
//!
//! Supports the split ONNX export published by andrewleech/qwen3-asr-*-onnx:
//! encoder, decoder_init, decoder_step, shared decoder weights, tokenizer.json,
//! and embed_tokens.bin. The int4 decoder files are preferred by default.

use super::whisper_mel::WhisperMelExtractor;
use super::Transcriber;
use crate::config::{Qwen3AsrConfig, Qwen3AsrQuantization};
use crate::error::TranscribeError;
use crate::onnx_runtime;
use half::f16;
use ort::session::Session;
use ort::value::{DynTensor, DynValue, Shape, Tensor, TensorElementType, ValueType};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tokenizers::Tokenizer;

const IM_START_TOKEN_ID: i64 = 151644;
const IM_END_TOKEN_ID: i64 = 151645;
const AUDIO_START_TOKEN_ID: i64 = 151669;
const AUDIO_END_TOKEN_ID: i64 = 151670;
const AUDIO_PAD_TOKEN_ID: i64 = 151676;
const ENDOFTEXT_TOKEN_ID: i64 = 151643;
const NEWLINE_TOKEN_ID: i64 = 198;
const SYSTEM_TOKEN_ID: i64 = 9125;
const USER_TOKEN_ID: i64 = 882;
const ASSISTANT_TOKEN_ID: i64 = 77091;

pub struct Qwen3AsrTranscriber {
    encoder: Mutex<Session>,
    decoder_init: Mutex<Session>,
    decoder_step: Mutex<Session>,
    tokenizer: Tokenizer,
    embeddings: Embeddings,
    mel_extractor: WhisperMelExtractor,
    max_tokens: usize,
    init_uses_input_embeds: bool,
    init_uses_past_kv: bool,
    decoder_kv_dtype: KvCacheDtype,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KvCacheDtype {
    F16,
    F32,
}

impl KvCacheDtype {
    fn from_tensor_type(ty: TensorElementType) -> Result<Self, TranscribeError> {
        match ty {
            TensorElementType::Float16 => Ok(Self::F16),
            TensorElementType::Float32 => Ok(Self::F32),
            other => Err(TranscribeError::InitFailed(format!(
                "Unsupported Qwen3-ASR decoder KV cache dtype '{}'. Expected f16 or f32.",
                other
            ))),
        }
    }

    fn tensor_element_type(self) -> TensorElementType {
        match self {
            Self::F16 => TensorElementType::Float16,
            Self::F32 => TensorElementType::Float32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KvCacheMetadata {
    dtype: KvCacheDtype,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Qwen3AsrTranscriber {
    pub fn new(config: &Qwen3AsrConfig) -> Result<Self, TranscribeError> {
        let model_dir = resolve_model_path(&config.model)?;
        let threads = config.threads.unwrap_or_else(|| num_cpus::get().min(4));

        tracing::info!(
            "Loading Qwen3-ASR model from {:?} (quantization={:?})",
            model_dir,
            config.quantization
        );
        let start = std::time::Instant::now();

        let encoder_file = resolve_onnx_file(&model_dir, "encoder", config.quantization)?;
        let decoder_init_file = resolve_onnx_file(&model_dir, "decoder_init", config.quantization)?;
        let decoder_step_file = resolve_onnx_file(&model_dir, "decoder_step", config.quantization)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(TranscribeError::ModelNotFound(format!(
                "Qwen3-ASR tokenizer not found: {}",
                tokenizer_path.display()
            )));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TranscribeError::InitFailed(format!("Failed to load tokenizer: {}", e)))?;

        let embeddings = Embeddings::load(&model_dir)?;

        let encoder = create_session(&encoder_file, threads, "Qwen3-ASR encoder")?;
        let decoder_init = create_session(&decoder_init_file, threads, "Qwen3-ASR decoder_init")?;
        let decoder_step = create_session(&decoder_step_file, threads, "Qwen3-ASR decoder_step")?;

        let init_input_names: Vec<String> = decoder_init
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let init_uses_input_embeds = if init_input_names.iter().any(|name| name == "input_ids") {
            false
        } else if init_input_names.iter().any(|name| name == "input_embeds") {
            true
        } else {
            return Err(TranscribeError::InitFailed(format!(
                "Unsupported Qwen3-ASR decoder_init inputs: {:?}. \
                 Expected input_ids or input_embeds based init.",
                init_input_names
            )));
        };

        let init_uses_past_kv = init_input_names.iter().any(|name| name == "past_keys")
            && init_input_names.iter().any(|name| name == "past_values");

        let step_kv_metadata = decoder_kv_metadata(&decoder_step, "decoder_step")?;
        let init_kv_metadata = if init_uses_past_kv {
            Some(decoder_kv_metadata(&decoder_init, "decoder_init")?)
        } else {
            None
        };

        if let Some(init_metadata) = init_kv_metadata {
            validate_kv_metadata_consistent(
                "decoder_init",
                init_metadata,
                "decoder_step",
                step_kv_metadata,
            )?;
        }

        let decoder_kv_dtype = init_kv_metadata.unwrap_or(step_kv_metadata).dtype;
        let (num_layers, num_kv_heads, head_dim) = init_kv_metadata
            .map(|metadata| {
                (
                    metadata.num_layers,
                    metadata.num_kv_heads,
                    metadata.head_dim,
                )
            })
            .unwrap_or((0, 0, 0));

        tracing::info!(
            "Qwen3-ASR model loaded in {:.2}s (encoder={}, init={}, step={}, embeddings={}x{}, init_format={}, past_kv={})",
            start.elapsed().as_secs_f32(),
            encoder_file.display(),
            decoder_init_file.display(),
            decoder_step_file.display(),
            embeddings.vocab_size(),
            embeddings.hidden_size(),
            if init_uses_input_embeds { "input_embeds" } else { "input_ids" },
            if init_uses_past_kv { "yes" } else { "no" },
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder_init: Mutex::new(decoder_init),
            decoder_step: Mutex::new(decoder_step),
            tokenizer,
            embeddings,
            mel_extractor: WhisperMelExtractor::new(128),
            max_tokens: config.max_tokens.max(1),
            init_uses_input_embeds,
            init_uses_past_kv,
            decoder_kv_dtype,
            num_layers,
            num_kv_heads,
            head_dim,
        })
    }

    fn prefill_chunk_size() -> usize {
        std::env::var("VOXTYPE_QWEN3_PREFILL_CHUNK_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(128)
    }

    fn empty_past_kv(&self) -> Result<(DynTensor, DynTensor), TranscribeError> {
        let shape = Shape::new([
            self.num_layers as i64,
            1,
            self.num_kv_heads as i64,
            0,
            self.head_dim as i64,
        ]);
        let dtype = self.decoder_kv_dtype.tensor_element_type();
        let empty_keys = DynTensor::new(&ort::memory::Allocator::default(), dtype, shape.clone())
            .map_err(|e| {
            TranscribeError::InferenceFailed(format!(
                "Failed to create empty past_keys tensor: {}",
                e
            ))
        })?;
        let empty_values = DynTensor::new(&ort::memory::Allocator::default(), dtype, shape)
            .map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Failed to create empty past_values tensor: {}",
                    e
                ))
            })?;

        Ok((empty_keys, empty_values))
    }

    fn run_encoder(&self, samples: &[f32]) -> Result<DynValue, TranscribeError> {
        let features = self.mel_extractor.extract(samples);
        let frames = features.len() / self.mel_extractor.n_mels();
        if frames == 0 {
            return Err(TranscribeError::AudioFormat(
                "Audio too short for Qwen3-ASR feature extraction".to_string(),
            ));
        }

        let mel_tensor =
            Tensor::<f32>::from_array(([1usize, self.mel_extractor.n_mels(), frames], features))
                .map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to create Qwen3-ASR mel tensor: {}",
                        e
                    ))
                })?;

        let mut encoder = self.encoder.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock Qwen3-ASR encoder: {}", e))
        })?;

        let mut outputs = encoder.run(ort::inputs![mel_tensor]).map_err(|e| {
            TranscribeError::InferenceFailed(format!("Qwen3-ASR encoder failed: {}", e))
        })?;

        outputs.remove("audio_features").ok_or_else(|| {
            TranscribeError::InferenceFailed(
                "Qwen3-ASR encoder did not produce audio_features".to_string(),
            )
        })
    }

    fn decode(&self, audio_features: &DynValue) -> Result<Vec<u32>, TranscribeError> {
        let (shape, _) = audio_features.try_extract_tensor::<f32>().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to inspect audio_features: {}", e))
        })?;
        let dims: &[i64] = shape;
        if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 {
            return Err(TranscribeError::InferenceFailed(format!(
                "Unexpected Qwen3-ASR audio_features shape: {:?}",
                dims
            )));
        }

        let audio_token_count = dims[1] as usize;
        let hidden_size = dims[2] as usize;
        let prompt_ids = build_prompt_ids(audio_token_count);
        let audio_start = audio_pad_start(&prompt_ids)?;
        let mut init = self.decoder_init.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!(
                "Failed to lock Qwen3-ASR decoder_init: {}",
                e
            ))
        })?;

        let (logits, mut present_keys, mut present_values) = if self.init_uses_past_kv
            && self.init_uses_input_embeds
        {
            let chunk_size = Self::prefill_chunk_size();

            let (_, audio_data) = audio_features.try_extract_tensor::<f32>().map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Failed to extract Qwen3-ASR audio features for chunked prefill: {}",
                    e
                ))
            })?;
            if hidden_size != self.embeddings.hidden_size() {
                return Err(TranscribeError::InferenceFailed(format!(
                    "Qwen3-ASR audio_features hidden_size {} does not match embeddings hidden_size {}",
                    hidden_size,
                    self.embeddings.hidden_size()
                )));
            }
            let embeds = build_prompt_embeds(
                &prompt_ids,
                audio_start,
                audio_token_count,
                audio_data,
                hidden_size,
                &self.embeddings,
            )?;

            let mut prev_present_keys: Option<DynValue> = None;
            let mut prev_present_values: Option<DynValue> = None;
            let mut last_logits: Option<DynValue> = None;

            for chunk_start in (0..prompt_ids.len()).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(prompt_ids.len());
                let chunk_len = chunk_end - chunk_start;

                let chunk_embeds =
                    embeds[chunk_start * hidden_size..chunk_end * hidden_size].to_vec();
                let chunk_position_ids: Vec<i64> = (chunk_start as i64..chunk_end as i64).collect();

                let chunk_embeds_tensor =
                    Tensor::<f32>::from_array(([1usize, chunk_len, hidden_size], chunk_embeds))
                        .map_err(|e| {
                            TranscribeError::InferenceFailed(format!(
                                "Failed to create Qwen3-ASR chunk input_embeds tensor: {}",
                                e
                            ))
                        })?;
                let chunk_pos_tensor =
                    Tensor::<i64>::from_array(([1usize, chunk_len], chunk_position_ids)).map_err(
                        |e| {
                            TranscribeError::InferenceFailed(format!(
                                "Failed to create Qwen3-ASR chunk position_ids tensor: {}",
                                e
                            ))
                        },
                    )?;

                let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
                    (
                        std::borrow::Cow::Borrowed("input_embeds"),
                        chunk_embeds_tensor.into(),
                    ),
                    (
                        std::borrow::Cow::Borrowed("position_ids"),
                        chunk_pos_tensor.into(),
                    ),
                ];

                match (&prev_present_keys, &prev_present_values) {
                    (Some(pk), Some(pv)) => {
                        inputs.push((
                            std::borrow::Cow::Borrowed("past_keys"),
                            ort::session::SessionInputValue::from(pk),
                        ));
                        inputs.push((
                            std::borrow::Cow::Borrowed("past_values"),
                            ort::session::SessionInputValue::from(pv),
                        ));
                    }
                    _ => {
                        let (empty_keys, empty_values) = self.empty_past_kv()?;
                        inputs.push((std::borrow::Cow::Borrowed("past_keys"), empty_keys.into()));
                        inputs.push((
                            std::borrow::Cow::Borrowed("past_values"),
                            empty_values.into(),
                        ));
                    }
                }

                let mut outputs = init.run(inputs).map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Qwen3-ASR decoder_init chunked call failed (chunk {}-{}): {}",
                        chunk_start, chunk_end, e
                    ))
                })?;

                prev_present_keys = Some(outputs.remove("present_keys").ok_or_else(|| {
                    TranscribeError::InferenceFailed(
                        "decoder_init did not produce present_keys".to_string(),
                    )
                })?);
                prev_present_values = Some(outputs.remove("present_values").ok_or_else(|| {
                    TranscribeError::InferenceFailed(
                        "decoder_init did not produce present_values".to_string(),
                    )
                })?);

                if chunk_end == prompt_ids.len() {
                    last_logits = Some(outputs.remove("logits").ok_or_else(|| {
                        TranscribeError::InferenceFailed(
                            "decoder_init did not produce logits".to_string(),
                        )
                    })?);
                }
            }

            (
                last_logits.ok_or_else(|| {
                    TranscribeError::InferenceFailed(
                        "Qwen3-ASR chunked prefill produced no logits".to_string(),
                    )
                })?,
                prev_present_keys.unwrap(),
                prev_present_values.unwrap(),
            )
        } else {
            let position_ids: Vec<i64> = (0..prompt_ids.len() as i64).collect();

            let position_ids_tensor =
                Tensor::<i64>::from_array(([1usize, position_ids.len()], position_ids)).map_err(
                    |e| {
                        TranscribeError::InferenceFailed(format!(
                            "Failed to create Qwen3-ASR position_ids tensor: {}",
                            e
                        ))
                    },
                )?;

            let init_inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
                if !self.init_uses_input_embeds {
                    let input_ids_tensor =
                        Tensor::<i64>::from_array(([1usize, prompt_ids.len()], prompt_ids.clone()))
                            .map_err(|e| {
                                TranscribeError::InferenceFailed(format!(
                                    "Failed to create Qwen3-ASR input_ids tensor: {}",
                                    e
                                ))
                            })?;
                    let audio_offset_tensor =
                        Tensor::<i64>::from_array(([1usize], vec![audio_start as i64])).map_err(
                            |e| {
                                TranscribeError::InferenceFailed(format!(
                                    "Failed to create Qwen3-ASR audio_offset tensor: {}",
                                    e
                                ))
                            },
                        )?;
                    let mut inputs = vec![
                        (
                            std::borrow::Cow::Borrowed("input_ids"),
                            input_ids_tensor.into(),
                        ),
                        (
                            std::borrow::Cow::Borrowed("position_ids"),
                            position_ids_tensor.into(),
                        ),
                        (
                            std::borrow::Cow::Borrowed("audio_features"),
                            ort::session::SessionInputValue::from(audio_features),
                        ),
                        (
                            std::borrow::Cow::Borrowed("audio_offset"),
                            audio_offset_tensor.into(),
                        ),
                    ];
                    if self.init_uses_past_kv {
                        let (empty_keys, empty_values) = self.empty_past_kv()?;
                        inputs.push((std::borrow::Cow::Borrowed("past_keys"), empty_keys.into()));
                        inputs.push((
                            std::borrow::Cow::Borrowed("past_values"),
                            empty_values.into(),
                        ));
                    }
                    inputs
                } else {
                    let (_, audio_data) =
                        audio_features.try_extract_tensor::<f32>().map_err(|e| {
                            TranscribeError::InferenceFailed(format!(
                        "Failed to extract Qwen3-ASR audio features for input_embeds init: {}",
                        e
                    ))
                        })?;
                    if hidden_size != self.embeddings.hidden_size() {
                        return Err(TranscribeError::InferenceFailed(format!(
                        "Qwen3-ASR audio_features hidden_size {} does not match embeddings hidden_size {}",
                        hidden_size,
                        self.embeddings.hidden_size()
                    )));
                    }
                    let embeds = build_prompt_embeds(
                        &prompt_ids,
                        audio_start,
                        audio_token_count,
                        audio_data,
                        hidden_size,
                        &self.embeddings,
                    )?;
                    let input_embeds_tensor = Tensor::<f32>::from_array((
                        [1usize, prompt_ids.len(), hidden_size],
                        embeds,
                    ))
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Failed to create Qwen3-ASR input_embeds tensor: {}",
                            e
                        ))
                    })?;
                    vec![
                        (
                            std::borrow::Cow::Borrowed("input_embeds"),
                            input_embeds_tensor.into(),
                        ),
                        (
                            std::borrow::Cow::Borrowed("position_ids"),
                            position_ids_tensor.into(),
                        ),
                    ]
                };

            let mut outputs = init.run(init_inputs).map_err(|e| {
                TranscribeError::InferenceFailed(format!("Qwen3-ASR decoder_init failed: {}", e))
            })?;

            let logits = outputs.remove("logits").ok_or_else(|| {
                TranscribeError::InferenceFailed("decoder_init did not produce logits".to_string())
            })?;
            let present_keys = outputs.remove("present_keys").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "decoder_init did not produce present_keys".to_string(),
                )
            })?;
            let present_values = outputs.remove("present_values").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "decoder_init did not produce present_values".to_string(),
                )
            })?;
            (logits, present_keys, present_values)
        };

        let mut next_token = argmax_last_logits(&logits)?;
        let mut generated = vec![next_token as u32];
        if is_eos(next_token) {
            return Ok(generated);
        }

        let mut step = self.decoder_step.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!(
                "Failed to lock Qwen3-ASR decoder_step: {}",
                e
            ))
        })?;

        let mut position = prompt_ids.len() as i64;
        for _ in 1..self.max_tokens {
            let token_embed = self.embeddings.lookup(next_token as usize)?;
            let token_embed_tensor = Tensor::<f32>::from_array((
                [1usize, 1usize, self.embeddings.hidden_size()],
                token_embed,
            ))
            .map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Failed to create Qwen3-ASR token embedding tensor: {}",
                    e
                ))
            })?;
            let step_pos_tensor = Tensor::<i64>::from_array(([1usize, 1usize], vec![position]))
                .map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to create Qwen3-ASR step position tensor: {}",
                        e
                    ))
                })?;

            let step_inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
                (
                    std::borrow::Cow::Borrowed("input_embeds"),
                    token_embed_tensor.into(),
                ),
                (
                    std::borrow::Cow::Borrowed("position_ids"),
                    step_pos_tensor.into(),
                ),
                (
                    std::borrow::Cow::Borrowed("past_keys"),
                    ort::session::SessionInputValue::from(&present_keys),
                ),
                (
                    std::borrow::Cow::Borrowed("past_values"),
                    ort::session::SessionInputValue::from(&present_values),
                ),
            ];

            let mut step_outputs = step.run(step_inputs).map_err(|e| {
                TranscribeError::InferenceFailed(format!("Qwen3-ASR decoder_step failed: {}", e))
            })?;

            let logits = step_outputs.remove("logits").ok_or_else(|| {
                TranscribeError::InferenceFailed("decoder_step did not produce logits".to_string())
            })?;
            present_keys = step_outputs.remove("present_keys").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "decoder_step did not produce present_keys".to_string(),
                )
            })?;
            present_values = step_outputs.remove("present_values").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "decoder_step did not produce present_values".to_string(),
                )
            })?;

            next_token = argmax_last_logits(&logits)?;
            generated.push(next_token as u32);
            position += 1;

            if is_eos(next_token) {
                break;
            }
        }

        Ok(generated)
    }
}

impl Transcriber for Qwen3AsrTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let audio_features = self.run_encoder(samples)?;
        let mut tokens = self.decode(&audio_features)?;
        while tokens.last().is_some_and(|&token| is_eos(token as i64)) {
            tokens.pop();
        }

        let text = self.tokenizer.decode(&tokens, true).map_err(|e| {
            TranscribeError::InferenceFailed(format!("Qwen3-ASR tokenizer decode failed: {}", e))
        })?;
        let text = text.trim().to_string();

        tracing::info!(
            "Qwen3-ASR transcription completed in {:.2}s: {:?}",
            start.elapsed().as_secs_f32(),
            if text.chars().count() > 50 {
                format!("{}...", text.chars().take(50).collect::<String>())
            } else {
                text.clone()
            }
        );

        Ok(text)
    }
}

fn create_session(
    path: &Path,
    threads: usize,
    component: &str,
) -> Result<Session, TranscribeError> {
    let builder = Session::builder()
        .map_err(|e| TranscribeError::InitFailed(format!("ONNX session builder failed: {}", e)))?
        .with_intra_threads(threads)
        .map_err(|e| TranscribeError::InitFailed(format!("Failed to set ONNX threads: {}", e)))?;

    let mut builder = onnx_runtime::maybe_apply_cuda(builder, None, component).map_err(|e| {
        TranscribeError::InitFailed(format!("Failed to configure CUDA for {}: {}", component, e))
    })?;

    builder.commit_from_file(path).map_err(|e| {
        TranscribeError::InitFailed(format!(
            "Failed to load {} from {:?}: {}",
            component, path, e
        ))
    })
}

fn decoder_kv_metadata(
    session: &Session,
    component: &str,
) -> Result<KvCacheMetadata, TranscribeError> {
    let keys = kv_tensor_metadata(session, component, "past_keys")?;
    let values = kv_tensor_metadata(session, component, "past_values")?;
    validate_kv_metadata_consistent(
        &format!("{} past_keys", component),
        keys,
        &format!("{} past_values", component),
        values,
    )?;
    Ok(keys)
}

fn kv_tensor_metadata(
    session: &Session,
    component: &str,
    input_name: &str,
) -> Result<KvCacheMetadata, TranscribeError> {
    let input = session
        .inputs()
        .iter()
        .find(|input| input.name() == input_name)
        .ok_or_else(|| {
            TranscribeError::InitFailed(format!(
                "Qwen3-ASR {} missing required {} input",
                component, input_name
            ))
        })?;

    let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
        return Err(TranscribeError::InitFailed(format!(
            "Qwen3-ASR {} {} input is not a tensor",
            component, input_name
        )));
    };

    let dims: &[i64] = shape;
    if dims.len() != 5 || dims[0] <= 0 || dims[2] <= 0 || dims[4] <= 0 {
        return Err(TranscribeError::InitFailed(format!(
            "Qwen3-ASR {} {} has unsupported KV shape {:?}. Expected [layers, batch, kv_heads, past, head_dim].",
            component, input_name, dims
        )));
    }

    Ok(KvCacheMetadata {
        dtype: KvCacheDtype::from_tensor_type(*ty)?,
        num_layers: dims[0] as usize,
        num_kv_heads: dims[2] as usize,
        head_dim: dims[4] as usize,
    })
}

fn validate_kv_metadata_consistent(
    left_name: &str,
    left: KvCacheMetadata,
    right_name: &str,
    right: KvCacheMetadata,
) -> Result<(), TranscribeError> {
    if left == right {
        Ok(())
    } else {
        Err(TranscribeError::InitFailed(format!(
            "Qwen3-ASR decoder KV metadata mismatch between {} ({:?}) and {} ({:?})",
            left_name, left, right_name, right
        )))
    }
}

fn argmax_last_logits(logits: &DynValue) -> Result<i64, TranscribeError> {
    let (shape, data) = logits.try_extract_tensor::<f32>().map_err(|e| {
        TranscribeError::InferenceFailed(format!("Failed to extract Qwen3-ASR logits: {}", e))
    })?;
    let dims: &[i64] = shape;

    let vocab = match dims {
        [1, seq_len, vocab] if *seq_len > 0 && *vocab > 0 => {
            let vocab = *vocab as usize;
            let offset = (*seq_len as usize - 1) * vocab;
            &data[offset..offset + vocab]
        }
        [1, vocab] if *vocab > 0 => &data[..*vocab as usize],
        _ => {
            return Err(TranscribeError::InferenceFailed(format!(
                "Unexpected Qwen3-ASR logits shape: {:?}",
                dims
            )))
        }
    };

    vocab
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .ok_or_else(|| TranscribeError::InferenceFailed("Empty Qwen3-ASR logits".to_string()))
}

fn build_prompt_embeds(
    prompt_ids: &[i64],
    audio_start: usize,
    audio_token_count: usize,
    audio_data: &[f32],
    hidden_size: usize,
    embeddings: &Embeddings,
) -> Result<Vec<f32>, TranscribeError> {
    let mut embeds = Vec::with_capacity(prompt_ids.len() * hidden_size);
    for (i, &token) in prompt_ids.iter().enumerate() {
        if i >= audio_start && i < audio_start + audio_token_count {
            let offset = (i - audio_start) * hidden_size;
            embeds.extend_from_slice(&audio_data[offset..offset + hidden_size]);
        } else {
            let emb = embeddings.lookup(token as usize)?;
            embeds.extend_from_slice(&emb);
        }
    }
    Ok(embeds)
}

fn build_prompt_ids(audio_token_count: usize) -> Vec<i64> {
    let mut ids = vec![
        IM_START_TOKEN_ID,
        SYSTEM_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        USER_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        AUDIO_START_TOKEN_ID,
    ];
    ids.extend(std::iter::repeat(AUDIO_PAD_TOKEN_ID).take(audio_token_count));
    ids.extend([
        AUDIO_END_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        ASSISTANT_TOKEN_ID,
        NEWLINE_TOKEN_ID,
    ]);
    ids
}

fn audio_pad_start(prompt_ids: &[i64]) -> Result<usize, TranscribeError> {
    prompt_ids
        .iter()
        .position(|&token| token == AUDIO_PAD_TOKEN_ID)
        .ok_or_else(|| {
            TranscribeError::InferenceFailed(
                "Qwen3-ASR prompt did not contain audio pad tokens".to_string(),
            )
        })
}

fn is_eos(token: i64) -> bool {
    token == ENDOFTEXT_TOKEN_ID || token == IM_END_TOKEN_ID
}

fn resolve_onnx_file(
    model_dir: &Path,
    name: &str,
    quantization: Qwen3AsrQuantization,
) -> Result<PathBuf, TranscribeError> {
    if matches!(quantization, Qwen3AsrQuantization::Int4) {
        let int4 = model_dir.join(format!("{}.int4.onnx", name));
        if int4.exists() {
            return Ok(int4);
        }
        tracing::warn!(
            "Qwen3-ASR int4 file not found for {}, falling back to FP32",
            name
        );
    }

    let full = model_dir.join(format!("{}.onnx", name));
    if full.exists() {
        Ok(full)
    } else {
        Err(TranscribeError::ModelNotFound(format!(
            "Qwen3-ASR {} model not found in {}",
            name,
            model_dir.display()
        )))
    }
}

fn resolve_model_path(model: &str) -> Result<PathBuf, TranscribeError> {
    let path = PathBuf::from(model);
    if path.is_absolute() && path.exists() {
        return Ok(path);
    }

    let models_dir = crate::config::Config::models_dir();
    let candidates = [
        models_dir.join(model),
        PathBuf::from(model),
        PathBuf::from("models").join(model),
        models_dir.join(format!("qwen3-asr-{}", model)),
        PathBuf::from("models").join(format!("qwen3-asr-{}", model)),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(TranscribeError::ModelNotFound(format!(
        "Qwen3-ASR model '{}' not found. Looked in:\n  - {}\n  - {}\n  - {}\n\n\
         Download the 1.7B int4 ONNX model from: \
         https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx",
        model,
        models_dir.join(model).display(),
        PathBuf::from(model).display(),
        PathBuf::from("models").join(model).display(),
    )))
}

#[derive(Deserialize)]
struct QwenConfig {
    embed_tokens_shape: Option<Vec<usize>>,
    embed_tokens_dtype: Option<String>,
    decoder: Option<QwenDecoderConfig>,
}

#[derive(Deserialize)]
struct QwenDecoderConfig {
    vocab_size: usize,
    hidden_size: usize,
}

enum Embeddings {
    F16 {
        data: Vec<u16>,
        vocab_size: usize,
        hidden_size: usize,
    },
    F32 {
        data: Vec<f32>,
        vocab_size: usize,
        hidden_size: usize,
    },
}

impl Embeddings {
    fn load(model_dir: &Path) -> Result<Self, TranscribeError> {
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Failed to read Qwen3-ASR config {}: {}",
                config_path.display(),
                e
            ))
        })?;
        let config: QwenConfig = serde_json::from_str(&config_text).map_err(|e| {
            TranscribeError::InitFailed(format!("Failed to parse Qwen3-ASR config.json: {}", e))
        })?;

        let (vocab_size, hidden_size) = if let Some(shape) = config.embed_tokens_shape {
            if shape.len() != 2 {
                return Err(TranscribeError::InitFailed(format!(
                    "Invalid embed_tokens_shape in Qwen3-ASR config: {:?}",
                    shape
                )));
            }
            (shape[0], shape[1])
        } else if let Some(decoder) = config.decoder {
            (decoder.vocab_size, decoder.hidden_size)
        } else {
            return Err(TranscribeError::InitFailed(
                "Qwen3-ASR config missing decoder vocab_size/hidden_size".to_string(),
            ));
        };

        let embed_path = model_dir.join("embed_tokens.bin");
        let bytes = std::fs::read(&embed_path).map_err(|e| {
            TranscribeError::ModelNotFound(format!(
                "Qwen3-ASR embed_tokens.bin not found or unreadable at {}: {}",
                embed_path.display(),
                e
            ))
        })?;

        let dtype = config
            .embed_tokens_dtype
            .unwrap_or_else(|| "float32".to_string())
            .to_lowercase();
        let values = vocab_size.checked_mul(hidden_size).ok_or_else(|| {
            TranscribeError::InitFailed("Qwen3-ASR embedding shape overflows usize".to_string())
        })?;

        match dtype.as_str() {
            "float16" | "fp16" => {
                let expected = values * 2;
                if bytes.len() != expected {
                    return Err(TranscribeError::InitFailed(format!(
                        "Qwen3-ASR embed_tokens.bin size mismatch: got {} bytes, expected {}",
                        bytes.len(),
                        expected
                    )));
                }
                let data = bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                Ok(Self::F16 {
                    data,
                    vocab_size,
                    hidden_size,
                })
            }
            "float32" | "fp32" => {
                let expected = values * 4;
                if bytes.len() != expected {
                    return Err(TranscribeError::InitFailed(format!(
                        "Qwen3-ASR embed_tokens.bin size mismatch: got {} bytes, expected {}",
                        bytes.len(),
                        expected
                    )));
                }
                let data = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Self::F32 {
                    data,
                    vocab_size,
                    hidden_size,
                })
            }
            other => Err(TranscribeError::InitFailed(format!(
                "Unsupported Qwen3-ASR embed_tokens_dtype '{}'",
                other
            ))),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::F16 { vocab_size, .. } | Self::F32 { vocab_size, .. } => *vocab_size,
        }
    }

    fn hidden_size(&self) -> usize {
        match self {
            Self::F16 { hidden_size, .. } | Self::F32 { hidden_size, .. } => *hidden_size,
        }
    }

    fn lookup(&self, token: usize) -> Result<Vec<f32>, TranscribeError> {
        if token >= self.vocab_size() {
            return Err(TranscribeError::InferenceFailed(format!(
                "Qwen3-ASR token {} is outside embedding vocab {}",
                token,
                self.vocab_size()
            )));
        }

        let hidden = self.hidden_size();
        let start = token * hidden;
        let end = start + hidden;
        match self {
            Self::F16 { data, .. } => Ok(data[start..end]
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect()),
            Self::F32 { data, .. } => Ok(data[start..end].to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_prompt_contains_requested_audio_tokens() {
        let prompt = build_prompt_ids(17);
        assert_eq!(
            prompt
                .iter()
                .filter(|&&token| token == AUDIO_PAD_TOKEN_ID)
                .count(),
            17
        );
        assert_eq!(audio_pad_start(&prompt).unwrap(), 9);
    }

    #[test]
    fn eos_ids_match_model_card() {
        assert!(is_eos(151643));
        assert!(is_eos(151645));
        assert!(!is_eos(151644));
    }

    #[test]
    fn build_prompt_embeds_replaces_audio_pads_with_features() {
        let hidden_size = 4usize;
        let vocab_size = 8usize;
        let mut embed_data = Vec::with_capacity(vocab_size * hidden_size);
        for token in 0..vocab_size {
            for h in 0..hidden_size {
                embed_data.push((token * 10 + h) as f32);
            }
        }
        let embeddings = Embeddings::F32 {
            data: embed_data,
            vocab_size,
            hidden_size,
        };

        let prompt = vec![0i64, 1, AUDIO_PAD_TOKEN_ID, AUDIO_PAD_TOKEN_ID, 2];
        let audio_start = 2usize;
        let audio_token_count = 2usize;
        let audio_data: Vec<f32> = vec![100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0];

        let result = build_prompt_embeds(
            &prompt,
            audio_start,
            audio_token_count,
            &audio_data,
            hidden_size,
            &embeddings,
        )
        .unwrap();

        assert_eq!(result.len(), prompt.len() * hidden_size);
        assert_eq!(&result[0..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&result[4..8], &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(&result[8..12], &[100.0, 101.0, 102.0, 103.0]);
        assert_eq!(&result[12..16], &[200.0, 201.0, 202.0, 203.0]);
        assert_eq!(&result[16..20], &[20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn build_prompt_embeds_without_audio_tokens() {
        let hidden_size = 3usize;
        let vocab_size = 4usize;
        let embed_data: Vec<f32> = (0..(vocab_size * hidden_size)).map(|i| i as f32).collect();
        let embeddings = Embeddings::F32 {
            data: embed_data,
            vocab_size,
            hidden_size,
        };

        let prompt = vec![0i64, 1, 2];
        let audio_data: Vec<f32> = vec![];

        let result =
            build_prompt_embeds(&prompt, 9, 0, &audio_data, hidden_size, &embeddings).unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(&result[0..3], &[0.0, 1.0, 2.0]);
        assert_eq!(&result[3..6], &[3.0, 4.0, 5.0]);
        assert_eq!(&result[6..9], &[6.0, 7.0, 8.0]);
    }
}
