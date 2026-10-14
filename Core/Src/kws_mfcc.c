/**
 * kws_mfcc.c - MFCC front end (CMSIS-DSP), matches python/kws_mfcc.py.
 *
 *  x[n]   = pcm[n] / 32768 * hann[n]          n < 640, zero-padded to 1024
 *  X      = arm_rfft_fast_f32(x)               (1024-point real FFT)
 *  P[k]   = Re^2 + Im^2
 *  mel[c] = sum sqrt(P[k]) * triangular weights (TF band-mapper form)
 *  mfcc   = DCT( log(max(mel, 1e-12)) )[0..9]
 */
#include "kws_mfcc.h"
#include "kws_mfcc_tables.h"
#include "arm_math.h"
#include <math.h>
#include <string.h>

static arm_rfft_fast_instance_f32 s_rfft;
static float s_fft_in[KWS_FFT_LEN];
static float s_fft_out[KWS_FFT_LEN];

int kws_mfcc_init(void)
{
    /* size-specific init: only the 1024-point tables get linked (saves ~100 KB flash) */
    return (arm_rfft_fast_init_1024_f32(&s_rfft) == ARM_MATH_SUCCESS) ? 0 : -1;
}

void kws_mfcc_frame(const int16_t *pcm, float *out)
{
    int i, k;
    float mel[KWS_NUM_MEL];

    /* 1) normalise to [-1,1) and window */
    for (i = 0; i < KWS_WIN_LEN; i++) {
        s_fft_in[i] = ((float)pcm[i] * (1.0f / 32768.0f)) * kws_hann[i];
    }
    memset(&s_fft_in[KWS_WIN_LEN], 0, sizeof(float) * (KWS_FFT_LEN - KWS_WIN_LEN));

    /* 2) real FFT. Output: [Re0, Re512, Re1, Im1, Re2, Im2, ...]
          (s_fft_in is used as scratch and destroyed) */
    arm_rfft_fast_f32(&s_rfft, s_fft_in, s_fft_out, 0);

    /* 3) magnitude -> mel filterbank (only bins 2..256 are non-zero) */
    memset(mel, 0, sizeof(mel));
    for (i = 0; i < KWS_MEL_NUM_BINS; i++) {
        const int bin = KWS_MEL_START_BIN + i;          /* 1 <= bin < 512 */
        const float re = s_fft_out[2 * bin];
        const float im = s_fft_out[2 * bin + 1];
        float mag;
        arm_sqrt_f32(re * re + im * im, &mag);
        const float wv = mag * kws_mel_weight[i];
        const int c = kws_mel_band[i];
        if (c >= 0)               mel[c]     += wv;
        if (c + 1 < KWS_NUM_MEL)  mel[c + 1] += mag - wv;
    }

    /* 4) log */
    for (i = 0; i < KWS_NUM_MEL; i++) {
        mel[i] = logf(mel[i] < 1e-12f ? 1e-12f : mel[i]);
    }

    /* 5) DCT-II -> 10 coefficients */
    for (k = 0; k < KWS_NUM_MFCC; k++) {
        float acc = 0.0f;
        for (i = 0; i < KWS_NUM_MEL; i++) {
            acc += kws_dct[k][i] * mel[i];
        }
        out[k] = acc;
    }
}

void kws_mfcc_quantize(const float *in, int8_t *out, int n, float scale, int zero_point)
{
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int32_t q = (int32_t)roundf(in[i] * inv) + zero_point;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}
