"""
kws_mfcc.py - PC-side REFERENCE of the MFCC front end (never runs on the STM32).

The STM32 runs ../c/kws_mfcc.c. This file exists only to:
  * convert DS_CNN_S.pb -> INT8 .tflite (needs features for calibration),
  * generate the const C tables (gen_c_tables.py),
  * prove C == Python == TensorFlow (test_mfcc_c_vs_py.py).

It is a numpy port of the TensorFlow kernels embedded in ARM's DS_CNN_S.pb:
    AudioSpectrogram(window_size=640, stride=320, magnitude_squared=True)
    Mfcc(lower=20 Hz, upper=4000 Hz, filterbank_channel_count=40, dct_coefficient_count=10)

Per 40 ms frame (640 samples, hop 320 = 20 ms, 49 frames per 1 s):
    x[n] = int16 / 32768 ; x[n] *= periodic Hann(640)
    zero-pad to 1024, real FFT -> 513 bins, power = re^2 + im^2
    mel: sqrt(power) weighted by 40 triangular filters (HTK mel, 20-4000 Hz)
    log(max(mel, 1e-12)) ; DCT-II (sqrt(2/40)), keep 10 coefficients
"""
import wave
import numpy as np

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
WIN_LEN = 640
HOP_LEN = 320
FFT_LEN = 1024
NUM_SPEC_BINS = FFT_LEN // 2 + 1
NUM_MEL = 40
NUM_MFCC = 10
MEL_LO_HZ = 20.0
MEL_HI_HZ = 4000.0
NUM_FRAMES = 1 + (CLIP_SAMPLES - WIN_LEN) // HOP_LEN   # 49
LOG_FLOOR = 1e-12
FEATURE_SIZE = NUM_FRAMES * NUM_MFCC                  # 490


def periodic_hann(n=WIN_LEN):
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / n)


def freq_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def mel_filterbank_tables():
    """Exact port of tensorflow/core/kernels/mfcc_mel_filterbank.cc.
    Returns (start_bin, end_bin, band_mapper[513], weights[513])."""
    mel_lo = freq_to_mel(MEL_LO_HZ)
    mel_hi = freq_to_mel(MEL_HI_HZ)
    spacing = (mel_hi - mel_lo) / (NUM_MEL + 1)
    centers = mel_lo + spacing * np.arange(1, NUM_MEL + 2, dtype=np.float64)
    hz_per_bin = 0.5 * SAMPLE_RATE / (NUM_SPEC_BINS - 1)
    start = int(1.5 + MEL_LO_HZ / hz_per_bin)
    end = int(MEL_HI_HZ / hz_per_bin)
    band = np.full(NUM_SPEC_BINS, -2, dtype=np.int32)
    weights = np.zeros(NUM_SPEC_BINS, dtype=np.float64)
    ch = 0
    for i in range(start, end + 1):
        melf = freq_to_mel(i * hz_per_bin)
        while ch < NUM_MEL and centers[ch] < melf:
            ch += 1
        band[i] = ch - 1
    for i in range(start, end + 1):
        melf = freq_to_mel(i * hz_per_bin)
        c = band[i]
        if c >= 0:
            weights[i] = (centers[c + 1] - melf) / (centers[c + 1] - centers[c])
        else:
            weights[i] = (centers[0] - melf) / (centers[0] - mel_lo)
    return start, end, band, weights


def dct_matrix():
    """tensorflow/core/kernels/mfcc_dct.cc : sqrt(2/N) * cos(pi/N * (j+0.5) * i)."""
    i = np.arange(NUM_MFCC, dtype=np.float64)[:, None]
    j = np.arange(NUM_MEL, dtype=np.float64)[None, :]
    return np.sqrt(2.0 / NUM_MEL) * np.cos(np.pi / NUM_MEL * (j + 0.5) * i)


_HANN = periodic_hann()
_START, _END, _BAND, _W = mel_filterbank_tables()
_DCT = dct_matrix()


def _dense_mel_matrix():
    """Same band-mapper weights as C, expanded to a dense (513, 40) matrix."""
    m = np.zeros((NUM_SPEC_BINS, NUM_MEL), dtype=np.float64)
    for i in range(_START, _END + 1):
        c = _BAND[i]
        if c >= 0:
            m[i, c] += _W[i]
        if c + 1 < NUM_MEL:
            m[i, c + 1] += 1.0 - _W[i]
    return m


_MELMAT = _dense_mel_matrix()


def mfcc_frames(frames):
    """(N, 640) float frames in [-1, 1) -> (N, 10) MFCCs (mirrors kws_mfcc_frame in C)."""
    x = np.zeros((frames.shape[0], FFT_LEN), dtype=np.float64)
    x[:, :WIN_LEN] = frames * _HANN
    spec = np.fft.rfft(x, axis=1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2)
    with np.errstate(all="ignore"):   # numpy2+Accelerate emits spurious matmul FP warnings
        mel = mag @ _MELMAT
        return np.log(np.maximum(mel, LOG_FLOOR)) @ _DCT.T


def mfcc_frame(frame):
    return mfcc_frames(np.asarray(frame, dtype=np.float64)[None, :])[0]


def pcm16_to_float(pcm):
    return np.asarray(pcm, dtype=np.float64) / 32768.0


def fit_to_clip(samples):
    out = np.zeros(CLIP_SAMPLES, dtype=samples.dtype)
    n = min(len(samples), CLIP_SAMPLES)
    out[:n] = samples[:n]
    return out


def mfcc_clip(samples_float):
    """1 s float audio -> (49, 10) MFCC matrix."""
    s = fit_to_clip(np.asarray(samples_float, dtype=np.float64))
    idx = np.arange(NUM_FRAMES)[:, None] * HOP_LEN + np.arange(WIN_LEN)[None, :]
    return mfcc_frames(s[idx])


def round_half_away(x):
    """Matches C roundf() and TFLite rounding."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize(feats, scale, zero_point):
    q = round_half_away(np.asarray(feats, dtype=np.float64) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def read_wav_int16(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16-bit PCM")
        rate, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2")
    if ch > 1:
        data = data.reshape(-1, ch)[:, 0]
    if rate != SAMPLE_RATE:
        t_new = np.arange(int(len(data) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE
        data = np.interp(t_new, np.arange(len(data)), data).astype(np.int16)
    return data


def wav_to_mfcc(path):
    return mfcc_clip(pcm16_to_float(read_wav_int16(path)))
