//! Cohere Transcribe ONNX transcription.
//!
//! Supports the INT8 split export published by tristanripke/cohere-transcribe-onnx-int8.
//! The encoder takes raw 16 kHz mono audio and emits decoder cross-attention K/V
//! caches; the decoder runs autoregressively from Cohere's prompt tokens.

use super::Transcriber;
use crate::config::CohereConfig;
use crate::error::TranscribeError;
use crate::onnx_runtime;
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const SAMPLE_RATE: usize = 16_000;
const CHUNK_SAMPLES: usize = SAMPLE_RATE * 30;
const STRIDE_SAMPLES: usize = SAMPLE_RATE * 25;
const NUM_DECODER_LAYERS: usize = 8;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const MAX_CONTEXT: usize = 1024;
const VOCAB_SIZE: usize = 16_384;
const CACHE_VALUES: usize = NUM_DECODER_LAYERS * NUM_HEADS * MAX_CONTEXT * HEAD_DIM;

pub struct CohereTranscriber {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokens: HashMap<i64, String>,
    token_to_id: HashMap<String, i64>,
    language: String,
    max_tokens: usize,
}

impl CohereTranscriber {
    pub fn new(config: &CohereConfig) -> Result<Self, TranscribeError> {
        let model_dir = resolve_model_path(&config.model)?;
        let threads = config.threads.unwrap_or_else(|| num_cpus::get().min(4));

        tracing::info!(
            "Loading Cohere Transcribe model from {:?} (language={})",
            model_dir,
            config.language
        );
        let start = std::time::Instant::now();

        let encoder_file = model_dir.join("cohere-encoder.int8.onnx");
        let decoder_file = model_dir.join("cohere-decoder.int8.onnx");
        let tokens_file = model_dir.join("tokens.txt");

        for required in [&encoder_file, &decoder_file, &tokens_file] {
            if !required.exists() {
                return Err(TranscribeError::ModelNotFound(format!(
                    "Cohere Transcribe model file not found: {}\n  \
                     Run 'voxtype setup model' to download Cohere Transcribe.",
                    required.display()
                )));
            }
        }

        let (tokens, token_to_id) = load_tokens(&tokens_file)?;
        let encoder = create_session(&encoder_file, threads, "Cohere encoder")?;
        let decoder = create_session(&decoder_file, threads, "Cohere decoder")?;

        let encoder_inputs: Vec<String> = encoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let decoder_inputs: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if !encoder_inputs.iter().any(|name| name == "audio")
            || !decoder_inputs.iter().any(|name| name == "tokens")
        {
            return Err(TranscribeError::InitFailed(format!(
                "Unsupported Cohere Transcribe ONNX contract. encoder inputs={:?}, decoder inputs={:?}",
                encoder_inputs, decoder_inputs
            )));
        }

        let language = normalize_language(&config.language);
        build_prompt_ids(&token_to_id, &language)?;

        tracing::info!(
            "Cohere Transcribe model loaded in {:.2}s (encoder={}, decoder={}, vocab={})",
            start.elapsed().as_secs_f32(),
            encoder_file.display(),
            decoder_file.display(),
            tokens.len(),
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokens,
            token_to_id,
            language,
            max_tokens: config.max_tokens.max(1).min(MAX_CONTEXT),
        })
    }

    fn transcribe_chunk(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        let (cross_k, cross_v) = self.run_encoder(samples)?;
        let token_ids = self.decode(&cross_k, &cross_v)?;
        Ok(decode_tokens(&token_ids, &self.tokens))
    }

    fn run_encoder(&self, samples: &[f32]) -> Result<(DynValue, DynValue), TranscribeError> {
        let audio_tensor = Tensor::<f32>::from_array(([1usize, samples.len()], samples.to_vec()))
            .map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to create Cohere audio tensor: {}", e))
        })?;

        let mut encoder = self.encoder.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock Cohere encoder: {}", e))
        })?;

        let inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> =
            vec![(std::borrow::Cow::Borrowed("audio"), audio_tensor.into())];
        let mut outputs = encoder.run(inputs).map_err(|e| {
            TranscribeError::InferenceFailed(format!("Cohere encoder failed: {}", e))
        })?;

        let cross_k = outputs.remove("n_layer_cross_k").ok_or_else(|| {
            TranscribeError::InferenceFailed(
                "Cohere encoder did not produce n_layer_cross_k".to_string(),
            )
        })?;
        let cross_v = outputs.remove("n_layer_cross_v").ok_or_else(|| {
            TranscribeError::InferenceFailed(
                "Cohere encoder did not produce n_layer_cross_v".to_string(),
            )
        })?;

        validate_cross_cache("n_layer_cross_k", &cross_k)?;
        validate_cross_cache("n_layer_cross_v", &cross_v)?;

        Ok((cross_k, cross_v))
    }

    fn decode(&self, cross_k: &DynValue, cross_v: &DynValue) -> Result<Vec<i64>, TranscribeError> {
        let prompt_ids = build_prompt_ids(&self.token_to_id, &self.language)?;
        let prompt_len = prompt_ids.len();
        let mut generated = prompt_ids.clone();
        let mut current_tokens = prompt_ids;
        let mut offset = 0i64;
        let eos_id = token_id(&self.token_to_id, "<|endoftext|>")?;

        let mut self_k_cache: Option<DynValue> = None;
        let mut self_v_cache: Option<DynValue> = None;

        let mut decoder = self.decoder.lock().map_err(|e| {
            TranscribeError::InferenceFailed(format!("Failed to lock Cohere decoder: {}", e))
        })?;

        for step in 0..self.max_tokens {
            if offset as usize + current_tokens.len() >= MAX_CONTEXT {
                tracing::warn!(
                    "Cohere decoder reached max context ({} tokens), stopping",
                    MAX_CONTEXT
                );
                break;
            }

            let token_count = current_tokens.len();
            let tokens_tensor =
                Tensor::<i64>::from_array(([1usize, token_count], current_tokens.clone()))
                    .map_err(|e| {
                        TranscribeError::InferenceFailed(format!(
                            "Failed to create Cohere tokens tensor: {}",
                            e
                        ))
                    })?;
            let offset_tensor = Tensor::<i64>::from_array(((), vec![offset])).map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Failed to create Cohere offset tensor: {}",
                    e
                ))
            })?;

            let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
                (std::borrow::Cow::Borrowed("tokens"), tokens_tensor.into()),
                (
                    std::borrow::Cow::Borrowed("n_layer_cross_k"),
                    ort::session::SessionInputValue::from(cross_k),
                ),
                (
                    std::borrow::Cow::Borrowed("n_layer_cross_v"),
                    ort::session::SessionInputValue::from(cross_v),
                ),
                (std::borrow::Cow::Borrowed("offset"), offset_tensor.into()),
            ];

            if let (Some(k_cache), Some(v_cache)) = (&self_k_cache, &self_v_cache) {
                inputs.push((
                    std::borrow::Cow::Borrowed("in_n_layer_self_k_cache"),
                    ort::session::SessionInputValue::from(k_cache),
                ));
                inputs.push((
                    std::borrow::Cow::Borrowed("in_n_layer_self_v_cache"),
                    ort::session::SessionInputValue::from(v_cache),
                ));
            } else {
                let zero_k = Tensor::<f32>::from_array((
                    [NUM_DECODER_LAYERS, 1usize, NUM_HEADS, MAX_CONTEXT, HEAD_DIM],
                    vec![0.0f32; CACHE_VALUES],
                ))
                .map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to create Cohere self K cache: {}",
                        e
                    ))
                })?;
                let zero_v = Tensor::<f32>::from_array((
                    [NUM_DECODER_LAYERS, 1usize, NUM_HEADS, MAX_CONTEXT, HEAD_DIM],
                    vec![0.0f32; CACHE_VALUES],
                ))
                .map_err(|e| {
                    TranscribeError::InferenceFailed(format!(
                        "Failed to create Cohere self V cache: {}",
                        e
                    ))
                })?;
                inputs.push((
                    std::borrow::Cow::Borrowed("in_n_layer_self_k_cache"),
                    zero_k.into(),
                ));
                inputs.push((
                    std::borrow::Cow::Borrowed("in_n_layer_self_v_cache"),
                    zero_v.into(),
                ));
            }

            let mut outputs = decoder.run(inputs).map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Cohere decoder failed at step {}: {}",
                    step, e
                ))
            })?;

            let next_token = {
                let logits = outputs.get("logits").ok_or_else(|| {
                    TranscribeError::InferenceFailed(
                        "Cohere decoder did not produce logits".to_string(),
                    )
                })?;
                argmax_last_logits(logits)?
            };

            if next_token == eos_id {
                break;
            }
            generated.push(next_token);

            self_k_cache = Some(outputs.remove("out_n_layer_self_k_cache").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "Cohere decoder did not produce out_n_layer_self_k_cache".to_string(),
                )
            })?);
            self_v_cache = Some(outputs.remove("out_n_layer_self_v_cache").ok_or_else(|| {
                TranscribeError::InferenceFailed(
                    "Cohere decoder did not produce out_n_layer_self_v_cache".to_string(),
                )
            })?);

            offset += token_count as i64;
            current_tokens = vec![next_token];
        }

        Ok(generated[prompt_len..].to_vec())
    }
}

impl Transcriber for CohereTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat(
                "Empty audio buffer".to_string(),
            ));
        }

        let start = std::time::Instant::now();
        let mut texts = Vec::new();

        if samples.len() <= CHUNK_SAMPLES {
            let text = self.transcribe_chunk(samples)?;
            if !text.is_empty() {
                texts.push(text);
            }
        } else {
            let mut chunk_start = 0usize;
            while chunk_start < samples.len() {
                let chunk_end = (chunk_start + CHUNK_SAMPLES).min(samples.len());
                let text = self.transcribe_chunk(&samples[chunk_start..chunk_end])?;
                if !text.is_empty() {
                    texts.push(text);
                }
                if chunk_end == samples.len() {
                    break;
                }
                chunk_start += STRIDE_SAMPLES;
            }
        }

        let text = texts.join(" ").trim().to_string();
        tracing::info!(
            "Cohere Transcribe completed in {:.2}s: {:?}",
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

fn argmax_last_logits(logits: &DynValue) -> Result<i64, TranscribeError> {
    let (shape, data) = logits.try_extract_tensor::<f32>().map_err(|e| {
        TranscribeError::InferenceFailed(format!("Failed to extract Cohere logits: {}", e))
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
                "Unexpected Cohere logits shape: {:?}",
                dims
            )))
        }
    };

    if vocab.len() != VOCAB_SIZE {
        tracing::debug!("Cohere logits vocab size is {}", vocab.len());
    }

    vocab
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .ok_or_else(|| TranscribeError::InferenceFailed("Empty Cohere logits".to_string()))
}

fn validate_cross_cache(name: &str, value: &DynValue) -> Result<(), TranscribeError> {
    let (shape, _) = value.try_extract_tensor::<f32>().map_err(|e| {
        TranscribeError::InferenceFailed(format!("Failed to inspect Cohere {}: {}", name, e))
    })?;
    let dims: &[i64] = shape;
    if dims.len() != 4 || dims[0] != NUM_DECODER_LAYERS as i64 || dims[1] != 1 || dims[3] <= 0 {
        return Err(TranscribeError::InferenceFailed(format!(
            "Unexpected Cohere {} shape: {:?}",
            name, dims
        )));
    }
    Ok(())
}

fn build_prompt_ids(
    token_to_id: &HashMap<String, i64>,
    language: &str,
) -> Result<Vec<i64>, TranscribeError> {
    let language_token = format!("<|{}|>", normalize_language(language));
    let prompt = [
        "<|startofcontext|>".to_string(),
        "<|startoftranscript|>".to_string(),
        "<|emo:undefined|>".to_string(),
        language_token.clone(),
        language_token,
        "<|pnc|>".to_string(),
        "<|noitn|>".to_string(),
        "<|notimestamp|>".to_string(),
        "<|nodiarize|>".to_string(),
    ];

    prompt
        .iter()
        .map(|token| token_id(token_to_id, token))
        .collect()
}

fn token_id(token_to_id: &HashMap<String, i64>, token: &str) -> Result<i64, TranscribeError> {
    token_to_id.get(token).copied().ok_or_else(|| {
        TranscribeError::InitFailed(format!(
            "Cohere Transcribe token '{}' not found in tokens.txt",
            token
        ))
    })
}

fn normalize_language(language: &str) -> String {
    language
        .trim()
        .trim_start_matches("<|")
        .trim_end_matches("|>")
        .to_lowercase()
}

fn decode_tokens(token_ids: &[i64], tokens: &HashMap<i64, String>) -> String {
    let mut text = String::new();
    let mut pending_bytes = Vec::new();

    for id in token_ids {
        let Some(token) = tokens.get(id) else {
            continue;
        };

        if let Some(byte) = parse_byte_token(token) {
            pending_bytes.push(byte);
            continue;
        }

        flush_bytes(&mut text, &mut pending_bytes);

        if token.starts_with("<|") || token == "<unk>" || token == "<pad>" {
            continue;
        }

        text.push_str(&token.replace('\u{2581}', " "));
    }

    flush_bytes(&mut text, &mut pending_bytes);
    text.trim().to_string()
}

fn parse_byte_token(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}

fn flush_bytes(text: &mut String, pending_bytes: &mut Vec<u8>) {
    if pending_bytes.is_empty() {
        return;
    }
    text.push_str(&String::from_utf8_lossy(pending_bytes));
    pending_bytes.clear();
}

fn load_tokens(
    path: &Path,
) -> Result<(HashMap<i64, String>, HashMap<String, i64>), TranscribeError> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        TranscribeError::InitFailed(format!(
            "Failed to read Cohere tokens.txt from {}: {}",
            path.display(),
            e
        ))
    })?;

    let mut tokens = HashMap::new();
    let mut token_to_id = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(last_space) = line.rfind(' ') {
            let token = line[..last_space].to_string();
            if let Ok(id) = line[last_space + 1..].parse::<i64>() {
                token_to_id.insert(token.clone(), id);
                tokens.insert(id, token);
            }
        }
    }

    if tokens.is_empty() {
        return Err(TranscribeError::InitFailed(
            "Cohere tokens.txt appears empty or malformed".to_string(),
        ));
    }

    Ok((tokens, token_to_id))
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
        models_dir.join("cohere-transcribe-onnx-int8"),
        PathBuf::from("models").join("cohere-transcribe-onnx-int8"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(TranscribeError::ModelNotFound(format!(
        "Cohere Transcribe model '{}' not found. Looked in:\n  - {}\n  - {}\n  - {}\n\n\
         Download the ONNX INT8 model from: \
         https://huggingface.co/tristanripke/cohere-transcribe-onnx-int8",
        model,
        models_dir.join(model).display(),
        PathBuf::from(model).display(),
        PathBuf::from("models").join(model).display(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_uses_language_twice() {
        let mut token_to_id = HashMap::new();
        for (idx, token) in [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|en|>",
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
        .iter()
        .enumerate()
        {
            token_to_id.insert((*token).to_string(), idx as i64);
        }

        let prompt = build_prompt_ids(&token_to_id, "en").unwrap();
        assert_eq!(prompt.len(), 9);
        assert_eq!(prompt.iter().filter(|&&id| id == 3).count(), 2);
    }

    #[test]
    fn decode_skips_specials_and_handles_sentencepiece() {
        let tokens = HashMap::from([
            (1, "<|startoftranscript|>".to_string()),
            (2, "\u{2581}hello".to_string()),
            (3, "world".to_string()),
            (4, "<0x21>".to_string()),
            (5, "<pad>".to_string()),
        ]);

        assert_eq!(decode_tokens(&[1, 2, 3, 4, 5], &tokens), "helloworld!");
    }

    #[test]
    fn normalizes_language_token_forms() {
        assert_eq!(normalize_language("EN"), "en");
        assert_eq!(normalize_language("<|de|>"), "de");
    }
}
