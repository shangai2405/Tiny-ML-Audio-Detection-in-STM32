/**
 * kws_mfcc.h - MFCC front end for the DS-CNN-S keyword spotter (STM32 / CMSIS-DSP).
 *
 * Bit-exact (to float32 precision) with python/kws_mfcc.py and with the
 * TensorFlow AudioSpectrogram+Mfcc ops used to train ARM's DS_CNN_S model.
 *
 * RAM use: 8 KB (two 1024-float FFT buffers). All tables are const (flash).
 */
#ifndef KWS_MFCC_H
#define KWS_MFCC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_SAMPLE_RATE   16000
#define KWS_WIN_LEN       640     /* 40 ms */
#define KWS_HOP_LEN       320     /* 20 ms */
#define KWS_FFT_LEN       1024
#define KWS_NUM_MEL       40
#define KWS_NUM_MFCC      10
#define KWS_NUM_FRAMES    49      /* 1 s window */
#define KWS_FEATURE_SIZE  (KWS_NUM_FRAMES * KWS_NUM_MFCC)  /* 490 = model input */

/** Initialise the CMSIS-DSP RFFT instance. Returns 0 on success. */
int  kws_mfcc_init(void);

/**
 * Compute 10 MFCCs from one 40 ms frame.
 * @param pcm  KWS_WIN_LEN int16 samples (contiguous)
 * @param out  KWS_NUM_MFCC float coefficients
 */
void kws_mfcc_frame(const int16_t *pcm, float *out);

/**
 * Quantise float MFCCs to the model's int8 input:
 *   q = clamp(roundf(x / scale) + zero_point, -128, 127)
 */
void kws_mfcc_quantize(const float *in, int8_t *out, int n,
                       float scale, int zero_point);

#ifdef __cplusplus
}
#endif
#endif /* KWS_MFCC_H */
