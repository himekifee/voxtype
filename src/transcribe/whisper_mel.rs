//! Whisper-compatible log-mel spectrogram extraction.
//!
//! Matches the host-side preprocessing used by WhisperFeatureExtractor-style
//! ONNX models: 16 kHz mono audio, centered STFT with reflect padding,
//! Hann window, Slaney mel filters, log10 compression, and Whisper scaling.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const WHISPER_CHUNK_SAMPLES: usize = SAMPLE_RATE * 30;
pub const WHISPER_CHUNK_FRAMES: usize = 3000;

pub struct WhisperMelExtractor {
    n_mels: usize,
    mel_filterbank: Vec<Vec<f32>>,
    hann_window: Vec<f32>,
}

impl WhisperMelExtractor {
    pub fn new(n_mels: usize) -> Self {
        Self {
            n_mels,
            mel_filterbank: compute_mel_filterbank(n_mels),
            hann_window: hann_window(N_FFT),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn extract(&self, samples: &[f32]) -> Vec<f32> {
        let padded = reflect_pad(samples, N_FFT / 2);
        let raw_frames = if padded.len() >= N_FFT {
            (padded.len() - N_FFT) / HOP_LENGTH + 1
        } else {
            0
        };
        let frames = raw_frames.saturating_sub(1);
        self.extract_frames(&padded, frames)
    }

    pub fn extract_fixed(
        &self,
        samples: &[f32],
        target_samples: usize,
        target_frames: usize,
    ) -> Vec<f32> {
        let mut fixed = vec![0.0f32; target_samples];
        let copy_len = samples.len().min(target_samples);
        fixed[..copy_len].copy_from_slice(&samples[..copy_len]);

        let mut features = self.extract(&fixed);
        let current_frames = features.len() / self.n_mels;
        if current_frames == target_frames {
            return features;
        }

        let mut resized = vec![0.0f32; self.n_mels * target_frames];
        let frames_to_copy = current_frames.min(target_frames);
        let values_to_copy = frames_to_copy * self.n_mels;
        resized[..values_to_copy].copy_from_slice(&features[..values_to_copy]);
        features.clear();
        resized
    }

    fn extract_frames(&self, padded: &[f32], frames: usize) -> Vec<f32> {
        if frames == 0 {
            return Vec::new();
        }

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let bins = N_FFT / 2 + 1;
        let mut output = vec![0.0f32; self.n_mels * frames];
        let mut max_log = f32::NEG_INFINITY;

        for frame_idx in 0..frames {
            let start = frame_idx * HOP_LENGTH;
            let mut fft_input: Vec<Complex<f32>> = (0..N_FFT)
                .map(|i| Complex::new(padded[start + i] * self.hann_window[i], 0.0))
                .collect();

            fft.process(&mut fft_input);

            let power: Vec<f32> = fft_input[..bins].iter().map(|c| c.norm_sqr()).collect();
            for mel_idx in 0..self.n_mels {
                let mel_energy: f32 = self.mel_filterbank[mel_idx]
                    .iter()
                    .zip(power.iter())
                    .map(|(&weight, &p)| weight * p)
                    .sum();
                let log_value = mel_energy.max(1e-10).log10();
                output[mel_idx * frames + frame_idx] = log_value;
                max_log = max_log.max(log_value);
            }
        }

        let floor = max_log - 8.0;
        for value in &mut output {
            *value = (value.max(floor) + 4.0) / 4.0;
        }

        output
    }
}

fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; pad * 2];
    }
    if samples.len() <= pad {
        let mut out = vec![0.0; pad];
        out.extend_from_slice(samples);
        out.extend(std::iter::repeat(0.0).take(pad));
        return out;
    }

    let mut out = Vec::with_capacity(samples.len() + pad * 2);
    for i in (1..=pad).rev() {
        out.push(samples[i]);
    }
    out.extend_from_slice(samples);
    for i in 0..pad {
        out.push(samples[samples.len() - 2 - i]);
    }
    out
}

fn hann_window(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos())
        .collect()
}

fn compute_mel_filterbank(n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(8000.0);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect();

    let mut filters = vec![vec![0.0; n_freqs]; n_mels];
    for mel_idx in 0..n_mels {
        let lower = hz_points[mel_idx];
        let center = hz_points[mel_idx + 1];
        let upper = hz_points[mel_idx + 2];
        let enorm = 2.0 / (upper - lower);

        for (bin_idx, &freq) in fft_freqs.iter().enumerate() {
            let lower_slope = (freq - lower) / (center - lower);
            let upper_slope = (upper - freq) / (upper - center);
            filters[mel_idx][bin_idx] = lower_slope.min(upper_slope).max(0.0) * enorm;
        }
    }

    filters
}

fn hz_to_mel(freq: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;

    if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        freq / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f32.ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * f_sp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thirty_second_chunk_has_whisper_frame_count() {
        let extractor = WhisperMelExtractor::new(80);
        let samples = vec![0.0; WHISPER_CHUNK_SAMPLES];
        let features = extractor.extract(&samples);
        assert_eq!(features.len(), 80 * WHISPER_CHUNK_FRAMES);
    }

    #[test]
    fn qwen_mel_uses_128_bins() {
        let extractor = WhisperMelExtractor::new(128);
        let samples = vec![0.0; SAMPLE_RATE];
        let features = extractor.extract(&samples);
        assert_eq!(features.len() % 128, 0);
        assert!(!features.is_empty());
    }
}
