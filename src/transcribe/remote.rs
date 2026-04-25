//! Remote speech-to-text transcription via OpenAI-compatible or Gemini API
//!
//! Supports:
//! - OpenAI-compatible endpoints (whisper.cpp server, OpenAI, etc.)
//! - Native Gemini API with inline base64 audio data
//!
//! Note: Remote APIs don't support language arrays. When a language array is
//! configured, the first/primary language is used.

use super::Transcriber;
use crate::config::{GeminiThinkingLevel, LanguageConfig, RemoteProvider, WhisperConfig};
use crate::error::TranscribeError;
use base64::{engine::general_purpose, Engine as _};
use chrono::{DateTime, Utc};
use std::cmp;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use ureq::serde_json;

const REMOTE_SAMPLE_RATE_HZ: u32 = 16_000;
const DEFAULT_REMOTE_RETRY_COUNT: u32 = 3;
const REMOTE_RETRY_INITIAL_DELAY: Duration = Duration::from_secs(1);
const REMOTE_RETRY_MAX_DELAY: Duration = Duration::from_secs(30);
const REMOTE_RETRY_LOG_BODY_LIMIT: usize = 240;
const GEMINI_MP3_MIME_TYPE: &str = "audio/mp3";
const GEMINI_WAV_MIME_TYPE: &str = "audio/wav";
const GEMINI_MP3_BITRATE_FFMPEG: &str = "32k";
const GEMINI_MP3_BITRATE_LAME: &str = "32";

struct GeminiAudioPayload {
    data: Vec<u8>,
    mime_type: &'static str,
    format_name: &'static str,
}

#[derive(Debug)]
enum RemoteRequestFailure {
    Status {
        code: u16,
        body: String,
        retry_after: Option<Duration>,
    },
    Transport {
        kind: ureq::ErrorKind,
        message: String,
    },
}

impl RemoteRequestFailure {
    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            Self::Transport { .. } => None,
        }
    }

    fn is_retryable(&self) -> bool {
        match self {
            Self::Status { code, .. } => RemoteTranscriber::is_retryable_status(*code),
            Self::Transport { kind, .. } => RemoteTranscriber::is_retryable_transport(*kind),
        }
    }

    fn summary(&self) -> String {
        match self {
            Self::Status { code, body, .. } => {
                let trimmed = body.trim();
                if trimmed.is_empty() {
                    format!("HTTP {}", code)
                } else {
                    format!(
                        "HTTP {}: {}",
                        code,
                        RemoteTranscriber::truncate_for_retry_log(
                            &RemoteTranscriber::sanitize_error_message(trimmed)
                        )
                    )
                }
            }
            Self::Transport { kind, message } => format!(
                "transport error ({:?}): {}",
                kind,
                RemoteTranscriber::truncate_for_retry_log(message)
            ),
        }
    }

    fn into_transcribe_error(self, provider: RemoteProvider) -> TranscribeError {
        match self {
            Self::Status { code, body, .. } => match provider {
                RemoteProvider::Gemini => TranscribeError::RemoteError(format!(
                    "Gemini server returned {}: {}",
                    code, body
                )),
                RemoteProvider::OpenAi => {
                    TranscribeError::RemoteError(format!("Server returned {}: {}", code, body))
                }
            },
            Self::Transport { message, .. } => match provider {
                RemoteProvider::Gemini => {
                    TranscribeError::NetworkError(format!("Gemini request failed: {}", message))
                }
                RemoteProvider::OpenAi => {
                    TranscribeError::NetworkError(format!("Request failed: {}", message))
                }
            },
        }
    }
}

/// Remote transcriber using OpenAI-compatible Whisper API or Gemini API
#[derive(Debug)]
pub struct RemoteTranscriber {
    /// Base endpoint URL (e.g., "http://192.168.1.100:8080")
    endpoint: String,
    /// Model name to send to server
    model: String,
    /// Language configuration
    language: LanguageConfig,
    /// Whether to translate to English
    translate: bool,
    /// Optional API key for authentication
    api_key: Option<String>,
    /// Optional initial prompt for transcription context
    initial_prompt: Option<String>,
    /// Optional Gemini thinking level
    gemini_thinking_level: Option<GeminiThinkingLevel>,
    /// Request timeout
    timeout: Duration,
    /// Retry attempts for transient remote failures
    retry_count: u32,
    /// Remote API provider
    provider: RemoteProvider,
}

impl RemoteTranscriber {
    /// Create a new remote transcriber from config
    pub fn new(config: &WhisperConfig) -> Result<Self, TranscribeError> {
        let endpoint = config
            .remote_endpoint
            .as_ref()
            .ok_or_else(|| {
                TranscribeError::ConfigError(
                    "remote_endpoint is required when mode = 'remote'".into(),
                )
            })?
            .clone();

        // Validate endpoint URL format
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            return Err(TranscribeError::ConfigError(format!(
                "remote_endpoint must start with http:// or https://, got: {}",
                endpoint
            )));
        }

        // Warn about non-HTTPS for non-localhost endpoints
        if endpoint.starts_with("http://")
            && !endpoint.contains("localhost")
            && !endpoint.contains("127.0.0.1")
            && !endpoint.contains("[::1]")
        {
            tracing::warn!(
                "Remote endpoint uses HTTP without TLS. Audio data will be transmitted unencrypted!"
            );
        }

        // Check for API key in config or environment
        let api_key = config
            .remote_api_key
            .clone()
            .or_else(|| std::env::var("VOXTYPE_REMOTE_API_KEY").ok())
            .or_else(|| std::env::var("VOXTYPE_WHISPER_API_KEY").ok());

        let provider = config.remote_provider.unwrap_or_default();

        let model = config
            .remote_model
            .clone()
            .unwrap_or_else(|| match provider {
                RemoteProvider::Gemini => "gemini-3-flash-preview".to_string(),
                RemoteProvider::OpenAi => "whisper-1".to_string(),
            });

        let timeout = Duration::from_secs(config.remote_timeout_secs.unwrap_or(30));
        let retry_count = config
            .remote_retry_count
            .unwrap_or(DEFAULT_REMOTE_RETRY_COUNT);

        // Warn if language array is configured (remote APIs don't support arrays)
        if config.language.is_multiple() {
            tracing::warn!(
                "Remote backend doesn't support language arrays. Using primary language '{}' from {:?}",
                config.language.primary(),
                config.language.as_vec()
            );
        }

        tracing::info!(
            "Configured remote transcriber: endpoint={}, model={}, provider={:?}, timeout={}s, retries={}",
            endpoint,
            model,
            provider,
            timeout.as_secs(),
            retry_count
        );

        let initial_prompt = config
            .initial_prompt
            .as_ref()
            .filter(|s| !s.is_empty())
            .cloned();

        Ok(Self {
            endpoint,
            model,
            language: config.language.clone(),
            translate: config.translate,
            api_key,
            initial_prompt,
            gemini_thinking_level: config.gemini_thinking_level,
            timeout,
            retry_count,
            provider,
        })
    }

    fn remote_failure_from_ureq(error: ureq::Error) -> RemoteRequestFailure {
        match error {
            ureq::Error::Status(code, response) => {
                let retry_after = response
                    .header("Retry-After")
                    .and_then(Self::parse_retry_after);
                let body = response.into_string().unwrap_or_default();
                RemoteRequestFailure::Status {
                    code,
                    body,
                    retry_after,
                }
            }
            ureq::Error::Transport(transport) => RemoteRequestFailure::Transport {
                kind: transport.kind(),
                message: Self::sanitize_error_message(&transport.to_string()),
            },
        }
    }

    fn sanitize_error_message(message: &str) -> String {
        let mut sanitized = String::with_capacity(message.len());
        let mut remaining = message;

        while let Some(relative_start) = Self::find_key_param(remaining) {
            sanitized.push_str(&remaining[..relative_start]);
            sanitized.push_str("key=[REDACTED]");

            let value_start = relative_start + "key=".len();
            let value = &remaining[value_start..];
            let value_end = value
                .find(|c: char| c == '&' || c == '#' || c == ':' || c.is_whitespace())
                .unwrap_or(value.len());
            remaining = &value[value_end..];
        }

        sanitized.push_str(remaining);
        sanitized
    }

    fn find_key_param(message: &str) -> Option<usize> {
        let mut search_from = 0;

        while let Some(relative_index) = message[search_from..].find("key=") {
            let index = search_from + relative_index;
            let is_param_boundary = index == 0
                || message[..index]
                    .chars()
                    .next_back()
                    .map(|c| c == '?' || c == '&' || c.is_whitespace())
                    .unwrap_or(false);

            if is_param_boundary {
                return Some(index);
            }

            search_from = index + "key=".len();
        }

        None
    }

    fn truncate_for_retry_log(message: &str) -> String {
        let mut chars = message.chars();
        let truncated: String = chars.by_ref().take(REMOTE_RETRY_LOG_BODY_LIMIT).collect();

        if chars.next().is_some() {
            format!("{}...", truncated)
        } else {
            truncated
        }
    }

    fn parse_retry_after(value: &str) -> Option<Duration> {
        let trimmed = value.trim();
        if let Ok(seconds) = trimmed.parse::<u64>() {
            return Some(Duration::from_secs(seconds));
        }

        DateTime::parse_from_rfc2822(trimmed)
            .ok()
            .and_then(|retry_at| {
                retry_at
                    .with_timezone(&Utc)
                    .signed_duration_since(Utc::now())
                    .to_std()
                    .ok()
            })
    }

    fn is_retryable_status(code: u16) -> bool {
        matches!(code, 408 | 429 | 500 | 502 | 503 | 504)
    }

    fn is_retryable_transport(kind: ureq::ErrorKind) -> bool {
        matches!(
            kind,
            ureq::ErrorKind::Dns
                | ureq::ErrorKind::ConnectionFailed
                | ureq::ErrorKind::Io
                | ureq::ErrorKind::ProxyConnect
        )
    }

    fn retry_delay(failed_attempt_index: u32, retry_after: Option<Duration>) -> Duration {
        let shift = failed_attempt_index.min(30);
        let multiplier = 1_u64.checked_shl(shift).unwrap_or(u64::MAX);
        let exponential = REMOTE_RETRY_INITIAL_DELAY.saturating_mul(multiplier as u32);
        let delay = retry_after.unwrap_or(exponential);
        cmp::min(delay, REMOTE_RETRY_MAX_DELAY)
    }

    fn send_remote_request_with_retries<F>(
        &self,
        operation: &str,
        send: F,
    ) -> Result<ureq::Response, TranscribeError>
    where
        F: FnMut() -> Result<ureq::Response, Box<ureq::Error>>,
    {
        self.send_remote_request_with_retries_and_sleep(operation, send, std::thread::sleep)
    }

    fn send_remote_request_with_retries_and_sleep<F, S>(
        &self,
        operation: &str,
        mut send: F,
        mut sleep: S,
    ) -> Result<ureq::Response, TranscribeError>
    where
        F: FnMut() -> Result<ureq::Response, Box<ureq::Error>>,
        S: FnMut(Duration),
    {
        for failed_attempts in 0..=self.retry_count {
            match send() {
                Ok(response) => return Ok(response),
                Err(error) => {
                    let failure = Self::remote_failure_from_ureq(*error);
                    if failed_attempts >= self.retry_count || !failure.is_retryable() {
                        return Err(failure.into_transcribe_error(self.provider));
                    }

                    let delay = Self::retry_delay(failed_attempts, failure.retry_after());
                    let summary = failure.summary();
                    tracing::warn!(
                        operation,
                        attempt = failed_attempts + 1,
                        max_retries = self.retry_count,
                        delay_ms = delay.as_millis(),
                        error = %summary,
                        "Transient remote request failure; retrying with exponential backoff"
                    );
                    sleep(delay);
                }
            }
        }

        unreachable!("retry loop always returns on success or final failure")
    }

    /// Encode f32 samples to WAV format
    fn encode_wav(&self, samples: &[f32]) -> Result<Vec<u8>, TranscribeError> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: REMOTE_SAMPLE_RATE_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut buffer, spec).map_err(|e| {
            TranscribeError::AudioFormat(format!("Failed to create WAV writer: {}", e))
        })?;

        // Convert f32 [-1.0, 1.0] to i16
        for &sample in samples {
            let clamped = sample.clamp(-1.0, 1.0);
            let scaled = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(scaled).map_err(|e| {
                TranscribeError::AudioFormat(format!("Failed to write sample: {}", e))
            })?;
        }

        writer
            .finalize()
            .map_err(|e| TranscribeError::AudioFormat(format!("Failed to finalize WAV: {}", e)))?;

        Ok(buffer.into_inner())
    }

    /// Encode WAV bytes to mono 16 kHz MP3 using an optional external encoder.
    fn encode_mp3_with_external_encoder(wav_data: &[u8]) -> Result<Vec<u8>, String> {
        Self::encode_mp3_with_external_encoder_paths(wav_data, "ffmpeg", "lame")
    }

    fn encode_mp3_with_external_encoder_paths(
        wav_data: &[u8],
        ffmpeg_path: &str,
        lame_path: &str,
    ) -> Result<Vec<u8>, String> {
        let temp_dir = tempfile::Builder::new()
            .prefix("voxtype_gemini_mp3_")
            .tempdir()
            .map_err(|e| format!("failed to create temp directory: {}", e))?;

        let wav_path = temp_dir.path().join("audio.wav");
        let mp3_path = temp_dir.path().join("audio.mp3");
        fs::write(&wav_path, wav_data).map_err(|e| format!("failed to write temp WAV: {}", e))?;

        let mut failures = Vec::new();

        let _ = fs::remove_file(&mp3_path);
        match Self::run_ffmpeg_mp3_encoder(ffmpeg_path, &wav_path, &mp3_path) {
            Ok(()) => return Self::read_mp3_output(&mp3_path),
            Err(err) => failures.push(format!("ffmpeg: {}", err)),
        }

        let _ = fs::remove_file(&mp3_path);
        match Self::run_lame_mp3_encoder(lame_path, &wav_path, &mp3_path) {
            Ok(()) => Self::read_mp3_output(&mp3_path),
            Err(err) => {
                failures.push(format!("lame: {}", err));
                Err(failures.join("; "))
            }
        }
    }

    fn run_ffmpeg_mp3_encoder(
        encoder_path: &str,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        let output = Command::new(encoder_path)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(input_path)
            .args([
                "-ac",
                "1",
                "-ar",
                "16000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                GEMINI_MP3_BITRATE_FFMPEG,
                "-f",
                "mp3",
            ])
            .arg(output_path)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            Ok(())
        } else {
            Err(Self::format_encoder_failure(
                output.status.code(),
                &output.stderr,
            ))
        }
    }

    fn run_lame_mp3_encoder(
        encoder_path: &str,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<(), String> {
        let output = Command::new(encoder_path)
            .args([
                "--quiet",
                "-m",
                "m",
                "--resample",
                "16",
                "-b",
                GEMINI_MP3_BITRATE_LAME,
            ])
            .arg(input_path)
            .arg(output_path)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| e.to_string())?;

        if output.status.success() {
            Ok(())
        } else {
            Err(Self::format_encoder_failure(
                output.status.code(),
                &output.stderr,
            ))
        }
    }

    fn read_mp3_output(path: &Path) -> Result<Vec<u8>, String> {
        let mp3_data = fs::read(path).map_err(|e| format!("failed to read MP3 output: {}", e))?;

        if mp3_data.is_empty() {
            Err("encoder produced an empty MP3 file".to_string())
        } else {
            Ok(mp3_data)
        }
    }

    fn format_encoder_failure(code: Option<i32>, stderr: &[u8]) -> String {
        let stderr = String::from_utf8_lossy(stderr);
        let stderr = stderr.trim();

        if stderr.is_empty() {
            format!("exited with code {:?}", code)
        } else {
            format!("exited with code {:?}: {}", code, stderr)
        }
    }

    /// Build Gemini audio bytes, preferring MP3 to reduce request size.
    fn encode_gemini_audio(&self, samples: &[f32]) -> Result<GeminiAudioPayload, TranscribeError> {
        let wav_data = self.encode_wav(samples)?;

        match Self::encode_mp3_with_external_encoder(&wav_data) {
            Ok(mp3_data) => {
                tracing::debug!(
                    wav_bytes = wav_data.len(),
                    mp3_bytes = mp3_data.len(),
                    "Encoded Gemini audio as MP3"
                );
                Ok(GeminiAudioPayload {
                    data: mp3_data,
                    mime_type: GEMINI_MP3_MIME_TYPE,
                    format_name: "MP3",
                })
            }
            Err(err) => {
                tracing::warn!(
                    "Could not encode Gemini audio as MP3 ({}); falling back to WAV. Install ffmpeg or lame to reduce Gemini request bandwidth.",
                    err
                );
                Ok(GeminiAudioPayload {
                    data: wav_data,
                    mime_type: GEMINI_WAV_MIME_TYPE,
                    format_name: "WAV",
                })
            }
        }
    }

    /// Build the multipart form body for the API request
    fn build_multipart_body(&self, wav_data: &[u8]) -> (String, Vec<u8>) {
        let boundary = format!(
            "----VoxtypeBoundary{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        let mut body = Vec::new();

        // Add file field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(
            b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(wav_data);
        body.extend_from_slice(b"\r\n");

        // Add model field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body.extend_from_slice(self.model.as_bytes());
        body.extend_from_slice(b"\r\n");

        // Add language field (if not auto-detect mode)
        // For language arrays, use the primary language since remote APIs don't support arrays
        if !self.language.is_auto() {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
            body.extend_from_slice(self.language.primary().as_bytes());
            body.extend_from_slice(b"\r\n");
        }

        // Add prompt field (if initial_prompt is configured)
        if let Some(ref prompt) = self.initial_prompt {
            body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"prompt\"\r\n\r\n");
            body.extend_from_slice(prompt.as_bytes());
            body.extend_from_slice(b"\r\n");
        }

        // Add response_format field
        body.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\n");
        body.extend_from_slice(b"json\r\n");

        // End boundary
        body.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

        (boundary, body)
    }

    /// Build the Gemini JSON request body
    fn build_gemini_request(&self, base64_audio: &str, mime_type: &str) -> serde_json::Value {
        let instruction = if self.translate {
            "Translate the audio into English.".to_string()
        } else {
            let mut text = "Transcribe the audio into text.".to_string();
            if !self.language.is_auto() {
                text.push_str(&format!(
                    " The spoken language is {}.",
                    self.language.primary()
                ));
            }
            text
        };

        let mut body = serde_json::json!({
            "contents": [{
                "parts": [
                    { "text": instruction },
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_audio
                        }
                    }
                ]
            }]
        });

        if let Some(ref prompt) = self.initial_prompt {
            if let serde_json::Value::Object(ref mut map) = body {
                map.insert(
                    "systemInstruction".to_string(),
                    serde_json::json!({
                        "parts": [{ "text": prompt }]
                    }),
                );
            }
        }

        if let Some(level) = self.gemini_thinking_level {
            if let serde_json::Value::Object(ref mut map) = body {
                map.insert(
                    "generationConfig".to_string(),
                    serde_json::json!({
                        "thinkingConfig": {
                            "thinkingLevel": level
                        }
                    }),
                );
            }
        }

        body
    }

    /// Parse the Gemini JSON response and extract transcript text
    fn parse_gemini_response(&self, json: &serde_json::Value) -> Result<String, TranscribeError> {
        json.get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.trim().to_string())
            .ok_or_else(|| {
                TranscribeError::RemoteError(format!(
                    "Gemini response missing expected fields: {}",
                    json
                ))
            })
    }

    fn transcribe_openai(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat("Empty audio buffer".into()));
        }

        let duration_secs = samples.len() as f32 / 16000.0;
        tracing::debug!(
            "Sending {:.2}s of audio to remote server ({} samples)",
            duration_secs,
            samples.len()
        );

        let start = std::time::Instant::now();

        // Encode audio to WAV
        let wav_data = self.encode_wav(samples)?;
        tracing::debug!("Encoded WAV: {} bytes", wav_data.len());

        // Build multipart form
        let (boundary, body) = self.build_multipart_body(&wav_data);

        // Determine the API path based on whether we're doing transcription or translation
        let path = if self.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };

        let url = format!("{}{}", self.endpoint.trim_end_matches('/'), path);

        // Send request, retrying transient server/network failures.
        let response = self.send_remote_request_with_retries("openai transcription", || {
            let mut request = ureq::post(&url).timeout(self.timeout).set(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", boundary),
            );

            if let Some(ref key) = self.api_key {
                request = request.set("Authorization", &format!("Bearer {}", key));
            }

            request.send_bytes(&body).map_err(Box::new)
        })?;

        // Parse JSON response
        let json: serde_json::Value = response.into_json().map_err(|e| {
            TranscribeError::RemoteError(format!("Failed to parse response: {}", e))
        })?;

        // Extract text from response
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                TranscribeError::RemoteError(format!("Response missing 'text' field: {}", json))
            })?
            .trim()
            .to_string();

        tracing::info!(
            "Remote transcription completed in {:.2}s: {:?}",
            start.elapsed().as_secs_f32(),
            if text.chars().count() > 50 {
                format!("{}...", text.chars().take(50).collect::<String>())
            } else {
                text.clone()
            }
        );

        Ok(text)
    }

    fn transcribe_gemini(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Err(TranscribeError::AudioFormat("Empty audio buffer".into()));
        }

        let duration_secs = samples.len() as f32 / 16000.0;
        tracing::debug!(
            "Sending {:.2}s of audio to Gemini ({} samples)",
            duration_secs,
            samples.len()
        );

        let start = std::time::Instant::now();

        let audio = self.encode_gemini_audio(samples)?;
        tracing::debug!(
            "Encoded Gemini audio as {}: {} bytes ({})",
            audio.format_name,
            audio.data.len(),
            audio.mime_type
        );

        let api_key = self.api_key.as_ref().ok_or_else(|| {
            TranscribeError::ConfigError("API key is required for Gemini remote provider".into())
        })?;

        let base64_audio = general_purpose::STANDARD.encode(&audio.data);

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.endpoint.trim_end_matches('/'),
            self.model,
            api_key
        );

        let body = self.build_gemini_request(&base64_audio, audio.mime_type);

        let body_string = body.to_string();

        let response = self.send_remote_request_with_retries("gemini generateContent", || {
            ureq::post(&url)
                .timeout(self.timeout)
                .set("Content-Type", "application/json")
                .send_string(&body_string)
                .map_err(Box::new)
        })?;

        let json: serde_json::Value = response.into_json().map_err(|e| {
            TranscribeError::RemoteError(format!("Failed to parse Gemini response: {}", e))
        })?;

        let text = self.parse_gemini_response(&json)?;

        tracing::info!(
            "Gemini transcription completed in {:.2}s: {:?}",
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

impl Transcriber for RemoteTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        match self.provider {
            RemoteProvider::Gemini => self.transcribe_gemini(samples),
            RemoteProvider::OpenAi => self.transcribe_openai(samples),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wav_basic() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        // Create a simple sine wave
        let samples: Vec<f32> = (0..16000)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin() * 0.5)
            .collect();

        let wav = transcriber.encode_wav(&samples).unwrap();

        // WAV header is 44 bytes, then 16000 samples * 2 bytes = 32000 bytes
        assert_eq!(wav.len(), 44 + 32000);

        // Check WAV magic
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn test_config_validation_missing_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: None,
            ..Default::default()
        };

        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("remote_endpoint"));
    }

    #[test]
    fn test_config_validation_invalid_url() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("not-a-url".to_string()),
            ..Default::default()
        };

        let result = RemoteTranscriber::new(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("http://"));
    }

    #[test]
    fn test_multipart_body_structure() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_model: Some("large-v3".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (boundary, body) = transcriber.build_multipart_body(&wav_data);

        let body_str = String::from_utf8_lossy(&body);

        // Verify boundary is used
        assert!(body_str.contains(&boundary));

        // Verify required fields
        assert!(body_str.contains("name=\"file\""));
        assert!(body_str.contains("filename=\"audio.wav\""));
        assert!(body_str.contains("Content-Type: audio/wav"));
        assert!(body_str.contains("name=\"model\""));
        assert!(body_str.contains("large-v3"));
        assert!(body_str.contains("name=\"language\""));
        assert!(body_str.contains("name=\"response_format\""));
        assert!(body_str.contains("json"));
    }

    #[test]
    fn test_multipart_body_includes_prompt() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: Some("Technical discussion about Rust and Kubernetes.".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("name=\"prompt\""));
        assert!(body_str.contains("Technical discussion about Rust and Kubernetes."));
    }

    #[test]
    fn test_multipart_body_excludes_empty_prompt() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: Some("".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(!body_str.contains("name=\"prompt\""));
    }

    #[test]
    fn test_multipart_body_excludes_prompt_when_none() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            initial_prompt: None,
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let wav_data = vec![0u8; 100];

        let (_boundary, body) = transcriber.build_multipart_body(&wav_data);
        let body_str = String::from_utf8_lossy(&body);

        assert!(!body_str.contains("name=\"prompt\""));
    }

    #[test]
    fn test_translate_false_uses_transcriptions_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            translate: false,
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        assert!(!transcriber.translate);
        let path = if transcriber.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };
        assert_eq!(path, "/v1/audio/transcriptions");
    }

    #[test]
    fn test_translate_true_uses_translations_endpoint() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            translate: true,
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        assert!(transcriber.translate);
        let path = if transcriber.translate {
            "/v1/audio/translations"
        } else {
            "/v1/audio/transcriptions"
        };
        assert_eq!(path, "/v1/audio/translations");
    }

    #[test]
    fn test_api_key_from_config() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_api_key: Some("sk-test-key-123".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.api_key, Some("sk-test-key-123".to_string()));
    }

    #[test]
    fn test_custom_timeout() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_timeout_secs: Some(60),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_default_timeout() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_custom_retry_count() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_retry_count: Some(5),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.retry_count, 5);
    }

    #[test]
    fn test_default_retry_count() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.retry_count, DEFAULT_REMOTE_RETRY_COUNT);
    }

    #[test]
    fn test_retryable_status_classification() {
        for code in [408, 429, 500, 502, 503, 504] {
            assert!(
                RemoteTranscriber::is_retryable_status(code),
                "status {} should retry",
                code
            );
        }

        for code in [400, 401, 403, 404, 422] {
            assert!(
                !RemoteTranscriber::is_retryable_status(code),
                "status {} should not retry",
                code
            );
        }
    }

    #[test]
    fn test_retryable_transport_classification() {
        for kind in [
            ureq::ErrorKind::Dns,
            ureq::ErrorKind::ConnectionFailed,
            ureq::ErrorKind::Io,
            ureq::ErrorKind::ProxyConnect,
        ] {
            assert!(
                RemoteTranscriber::is_retryable_transport(kind),
                "transport {:?} should retry",
                kind
            );
        }

        for kind in [
            ureq::ErrorKind::InvalidUrl,
            ureq::ErrorKind::UnknownScheme,
            ureq::ErrorKind::InsecureRequestHttpsOnly,
            ureq::ErrorKind::TooManyRedirects,
            ureq::ErrorKind::BadStatus,
            ureq::ErrorKind::BadHeader,
            ureq::ErrorKind::InvalidProxyUrl,
            ureq::ErrorKind::ProxyUnauthorized,
        ] {
            assert!(
                !RemoteTranscriber::is_retryable_transport(kind),
                "transport {:?} should not retry",
                kind
            );
        }
    }

    #[test]
    fn test_sanitize_error_message_redacts_gemini_key() {
        let sanitized = RemoteTranscriber::sanitize_error_message(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=secret-token: Connection Failed",
        );

        assert_eq!(
            sanitized,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent?key=[REDACTED]: Connection Failed"
        );
        assert!(!sanitized.contains("secret-token"));
    }

    #[test]
    fn test_sanitize_error_message_redacts_key_between_query_params() {
        let sanitized = RemoteTranscriber::sanitize_error_message(
            "https://example.test/path?alt=json&key=secret-token&pretty=false: Network Error",
        );

        assert_eq!(
            sanitized,
            "https://example.test/path?alt=json&key=[REDACTED]&pretty=false: Network Error"
        );
        assert!(!sanitized.contains("secret-token"));
    }

    #[test]
    fn test_truncate_for_retry_log_limits_long_messages() {
        let long_message = "x".repeat(REMOTE_RETRY_LOG_BODY_LIMIT + 10);
        let truncated = RemoteTranscriber::truncate_for_retry_log(&long_message);

        assert_eq!(truncated.chars().count(), REMOTE_RETRY_LOG_BODY_LIMIT + 3);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_remote_failure_summary_sanitizes_and_truncates_status_body() {
        let body = format!(
            "temporary failure for key=secret-token {}",
            "x".repeat(REMOTE_RETRY_LOG_BODY_LIMIT)
        );
        let failure = RemoteRequestFailure::Status {
            code: 503,
            body,
            retry_after: None,
        };
        let summary = failure.summary();

        assert!(summary.starts_with("HTTP 503: temporary failure for key=[REDACTED]"));
        assert!(summary.ends_with("..."));
        assert!(!summary.contains("secret-token"));
    }

    #[test]
    fn test_retry_after_seconds_parsing() {
        assert_eq!(
            RemoteTranscriber::parse_retry_after("7"),
            Some(Duration::from_secs(7))
        );
    }

    #[test]
    fn test_send_remote_request_retries_503_then_succeeds() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_retry_count: Some(3),
            ..Default::default()
        };
        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let mut attempts = 0;
        let mut sleeps = Vec::new();

        let response = transcriber
            .send_remote_request_with_retries_and_sleep(
                "test retry",
                || {
                    attempts += 1;
                    if attempts == 1 {
                        let response =
                            ureq::Response::new(503, "Service Unavailable", "busy").unwrap();
                        Err(Box::new(ureq::Error::Status(503, response)))
                    } else {
                        Ok(ureq::Response::new(200, "OK", "ok").unwrap())
                    }
                },
                |delay| sleeps.push(delay),
            )
            .unwrap();

        assert_eq!(attempts, 2);
        assert_eq!(sleeps, vec![Duration::from_secs(1)]);
        assert_eq!(response.status(), 200);
    }

    #[test]
    fn test_send_remote_request_retry_count_zero_disables_retries() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_retry_count: Some(0),
            ..Default::default()
        };
        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let mut attempts = 0;
        let mut sleeps = Vec::new();

        let error = transcriber
            .send_remote_request_with_retries_and_sleep(
                "test retry disabled",
                || {
                    attempts += 1;
                    let response = ureq::Response::new(503, "Service Unavailable", "busy").unwrap();
                    Err(Box::new(ureq::Error::Status(503, response)))
                },
                |delay| sleeps.push(delay),
            )
            .unwrap_err();

        assert_eq!(attempts, 1);
        assert!(sleeps.is_empty());
        assert!(error.to_string().contains("Server returned 503"));
    }

    #[test]
    fn test_send_remote_request_does_not_retry_non_retryable_status() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_retry_count: Some(3),
            ..Default::default()
        };
        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let mut attempts = 0;
        let mut sleeps = Vec::new();

        let error = transcriber
            .send_remote_request_with_retries_and_sleep(
                "test non-retryable",
                || {
                    attempts += 1;
                    let response = ureq::Response::new(401, "Unauthorized", "bad key").unwrap();
                    Err(Box::new(ureq::Error::Status(401, response)))
                },
                |delay| sleeps.push(delay),
            )
            .unwrap_err();

        assert_eq!(attempts, 1);
        assert!(sleeps.is_empty());
        assert!(error.to_string().contains("Server returned 401"));
    }

    #[test]
    fn test_send_remote_request_retries_until_limit() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            remote_retry_count: Some(2),
            ..Default::default()
        };
        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let mut attempts = 0;
        let mut sleeps = Vec::new();

        let error = transcriber
            .send_remote_request_with_retries_and_sleep(
                "test retry limit",
                || {
                    attempts += 1;
                    let response =
                        ureq::Response::new(503, "Service Unavailable", "still busy").unwrap();
                    Err(Box::new(ureq::Error::Status(503, response)))
                },
                |delay| sleeps.push(delay),
            )
            .unwrap_err();

        assert_eq!(attempts, 3);
        assert_eq!(sleeps, vec![Duration::from_secs(1), Duration::from_secs(2)]);
        assert!(error.to_string().contains("Server returned 503"));
    }

    #[test]
    fn test_retry_delay_uses_exponential_backoff_and_cap() {
        assert_eq!(
            RemoteTranscriber::retry_delay(0, None),
            Duration::from_secs(1)
        );
        assert_eq!(
            RemoteTranscriber::retry_delay(1, None),
            Duration::from_secs(2)
        );
        assert_eq!(
            RemoteTranscriber::retry_delay(5, None),
            Duration::from_secs(30)
        );
        assert_eq!(
            RemoteTranscriber::retry_delay(0, Some(Duration::from_secs(9))),
            Duration::from_secs(9)
        );
        assert_eq!(
            RemoteTranscriber::retry_delay(0, Some(Duration::from_secs(90))),
            REMOTE_RETRY_MAX_DELAY
        );
    }

    #[test]
    fn test_gemini_default_model() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.model, "gemini-3-flash-preview");
        assert!(matches!(transcriber.provider, RemoteProvider::Gemini));
    }

    #[test]
    fn test_openai_default_model() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("http://localhost:8080".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        assert_eq!(transcriber.model, "whisper-1");
        assert!(matches!(transcriber.provider, RemoteProvider::OpenAi));
    }

    #[test]
    fn test_encode_mp3_with_external_encoder() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let config = WhisperConfig {
                mode: Some(crate::config::WhisperMode::Remote),
                remote_endpoint: Some(
                    "https://generativelanguage.googleapis.com/v1beta".to_string(),
                ),
                remote_provider: Some(RemoteProvider::Gemini),
                ..Default::default()
            };

            let transcriber = RemoteTranscriber::new(&config).unwrap();
            let samples: Vec<f32> = (0..REMOTE_SAMPLE_RATE_HZ)
                .map(|i| {
                    (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / REMOTE_SAMPLE_RATE_HZ as f32)
                        .sin()
                        * 0.5
                })
                .collect();

            let wav = transcriber.encode_wav(&samples).unwrap();
            let temp_dir = tempfile::tempdir().unwrap();
            let fake_ffmpeg = temp_dir.path().join("fake-ffmpeg");
            let missing_lame = temp_dir.path().join("missing-lame");
            fs::write(
                &fake_ffmpeg,
                "#!/bin/sh\nfor out do :; done\nprintf 'ID3fake-mp3' > \"$out\"\n",
            )
            .unwrap();
            let mut permissions = fs::metadata(&fake_ffmpeg).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&fake_ffmpeg, permissions).unwrap();

            let mp3 = RemoteTranscriber::encode_mp3_with_external_encoder_paths(
                &wav,
                fake_ffmpeg.to_str().unwrap(),
                missing_lame.to_str().unwrap(),
            )
            .unwrap();

            assert_eq!(mp3, b"ID3fake-mp3");
        }
    }

    #[test]
    fn test_gemini_request_structure() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            language: LanguageConfig::Single("de".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let request = transcriber.build_gemini_request("dummyaudio", GEMINI_MP3_MIME_TYPE);

        let contents = request.get("contents").unwrap().as_array().unwrap();
        assert_eq!(contents.len(), 1);

        let parts = contents[0].get("parts").unwrap().as_array().unwrap();
        assert_eq!(parts.len(), 2);

        let text_part = &parts[0];
        assert_eq!(
            text_part.get("text").unwrap().as_str().unwrap(),
            "Transcribe the audio into text. The spoken language is de."
        );

        let audio_part = &parts[1];
        let inline_data = audio_part.get("inlineData").unwrap();
        assert_eq!(
            inline_data.get("mimeType").unwrap().as_str().unwrap(),
            "audio/mp3"
        );
        assert_eq!(
            inline_data.get("data").unwrap().as_str().unwrap(),
            "dummyaudio"
        );

        assert!(request.get("systemInstruction").is_none());
        assert!(request.get("generationConfig").is_none());
    }

    #[test]
    fn test_gemini_request_with_thinking_level() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            gemini_thinking_level: Some(GeminiThinkingLevel::Medium),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let request = transcriber.build_gemini_request("dummyaudio", GEMINI_MP3_MIME_TYPE);

        assert_eq!(
            request
                .get("generationConfig")
                .and_then(|v| v.get("thinkingConfig"))
                .and_then(|v| v.get("thinkingLevel"))
                .and_then(|v| v.as_str()),
            Some("medium")
        );
    }

    #[test]
    fn test_gemini_request_with_translate() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            translate: true,
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let request = transcriber.build_gemini_request("dummyaudio", GEMINI_MP3_MIME_TYPE);

        let parts = request.get("contents").unwrap().as_array().unwrap()[0]
            .get("parts")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(
            parts[0].get("text").unwrap().as_str().unwrap(),
            "Translate the audio into English."
        );
    }

    #[test]
    fn test_gemini_request_with_initial_prompt() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            initial_prompt: Some("Medical terminology.".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let request = transcriber.build_gemini_request("dummyaudio", GEMINI_MP3_MIME_TYPE);

        let system = request.get("systemInstruction").unwrap();
        let system_parts = system.get("parts").unwrap().as_array().unwrap();
        assert_eq!(
            system_parts[0].get("text").unwrap().as_str().unwrap(),
            "Medical terminology."
        );
    }

    #[test]
    fn test_gemini_request_with_auto_language() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            language: LanguageConfig::Single("auto".to_string()),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();
        let request = transcriber.build_gemini_request("dummyaudio", GEMINI_MP3_MIME_TYPE);

        let parts = request.get("contents").unwrap().as_array().unwrap()[0]
            .get("parts")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(
            parts[0].get("text").unwrap().as_str().unwrap(),
            "Transcribe the audio into text."
        );
    }

    #[test]
    fn test_gemini_response_parsing() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        let response = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "  Hello world  " }],
                    "role": "model"
                }
            }]
        });

        let text = transcriber.parse_gemini_response(&response).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_gemini_response_missing_fields_returns_error() {
        let config = WhisperConfig {
            mode: Some(crate::config::WhisperMode::Remote),
            remote_endpoint: Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            remote_provider: Some(RemoteProvider::Gemini),
            ..Default::default()
        };

        let transcriber = RemoteTranscriber::new(&config).unwrap();

        let response = serde_json::json!({ "error": { "message": "bad request" } });
        assert!(transcriber.parse_gemini_response(&response).is_err());
    }
}
