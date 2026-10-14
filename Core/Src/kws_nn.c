/**
 * kws_nn.c - int8 DS-CNN-S inference (see kws_nn.h). Plain C + Cortex-M4 DSP intrinsics
 * (SMLAD) for the 1x1 convolutions; falls back to portable emulation on a PC (KWS_NN_HOST).
 */
#include "kws_nn.h"
#include <math.h>

#ifdef KWS_NN_HOST
/* portable emulation of the few Cortex-M4 DSP instructions used below */
static inline uint32_t __SXTB16(uint32_t x)
{
    uint16_t lo = (uint16_t)(int16_t)(int8_t)(x & 0xFFu);
    uint16_t hi = (uint16_t)(int16_t)(int8_t)((x >> 16) & 0xFFu);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
static inline int32_t __SMLAD(uint32_t x, uint32_t y, int32_t acc)
{
    return acc + (int32_t)(int16_t)x * (int16_t)y + (int32_t)(int16_t)(x >> 16) * (int16_t)(y >> 16);
}
static inline uint32_t __ROR(uint32_t x, uint32_t n) { return (x >> n) | (x << (32u - n)); }
int8_t kws_nn_dbg[11][8000];
#define DBG(i, p, n) do { for (int _k = 0; _k < (n); _k++) kws_nn_dbg[i][_k] = (p)[_k]; } while (0)
#else
#include "stm32f4xx.h"          /* CMSIS core: __SXTB16, __SMLAD, __ROR */
#define DBG(i, p, n) do { } while (0)
#endif

#define H   25
#define W   5
#define C   64
#define NPOS (H * W)            /* 125 */

static int8_t s_bufA[NPOS * C] __attribute__((aligned(4)));   /* 8000 B */
static int8_t s_bufB[NPOS * C] __attribute__((aligned(4)));   /* 8000 B */

/* ---- TFLite fixed-point requantisation ---------------------------------------------- */
static inline int32_t srdhm(int32_t a, int32_t b)    /* SaturatingRoundingDoublingHighMul */
{
    /* (a == b == INT32_MIN would saturate, but every multiplier here is positive) */
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));
}

static inline int32_t rdpot(int32_t x, int exp)       /* RoundingDivideByPOT: round half away from zero */
{
    if (exp == 0) return x;
    /* same result as gemmlowp's mask/remainder/threshold form: add (half - 1) for negatives, half for
       positives, then an arithmetic shift  (x >> 31 is -1 for negative x, 0 otherwise) */
    return (x + (1 << (exp - 1)) + (x >> 31)) >> exp;
}

static inline int32_t requant(int32_t acc, int32_t mult, int shift)   /* MultiplyByQuantizedMultiplier */
{
    int ls = shift > 0 ? shift : 0;
    int rs = shift > 0 ? 0 : -shift;
    return rdpot(srdhm(acc * (1 << ls), mult), rs);
}

static inline int8_t sat8_relu(int32_t v, int32_t zp)  /* + output zero point, fused ReLU lower bound = zp */
{
    v += zp;
    if (v < zp)  v = zp;          /* ReLU: real 0 == zp (for these layers zp = -128, same as int8 min) */
    if (v > 127) v = 127;
    return (int8_t)v;
}

/* ---- conv_1: 10x4 kernel, stride 2x2, SAME, 1 -> 64 channels --------------------------- */
/* Each output position needs a 40-tap dot product per filter.  The 40 inputs (minus the input
   zero point, padding = 0) are packed once into 20 words of int16 lane pairs; the weights are
   stored the same way (kws_c1_wp), so every SMLAD does two taps.  4 filters per pass share the
   input loads. */
static void conv1(const int8_t *in, int8_t *out)
{
    int16_t x[49 * 10];                               /* input minus zero point */
    for (int i = 0; i < 490; i++) x[i] = (int16_t)(in[i] - KWS_C1_IN_ZP);

    for (int oy = 0; oy < 25; oy++) {
        const int iy0 = oy * 2 - 4;                   /* SAME padding: top 4 */
        for (int ox = 0; ox < 5; ox++) {
            const int ix0 = ox * 2 - 1;               /* SAME padding: left 1 */
            uint32_t xv[20];
            for (int ky = 0; ky < 10; ky++) {
                const int iy = iy0 + ky;
                for (int kp = 0; kp < 2; kp++) {
                    const int ix = ix0 + 2 * kp;
                    const int ok_y = (iy >= 0 && iy < 49);
                    const int16_t v0 = (ok_y && ix >= 0 && ix < 10)         ? x[iy * 10 + ix]     : 0;
                    const int16_t v1 = (ok_y && ix + 1 >= 0 && ix + 1 < 10) ? x[iy * 10 + ix + 1] : 0;
                    xv[ky * 2 + kp] = (uint32_t)(uint16_t)v0 | ((uint32_t)(uint16_t)v1 << 16);
                }
            }
            int8_t *o = out + (oy * 5 + ox) * C;
            for (int oc = 0; oc < C; oc += 4) {
                const uint32_t *w0 = kws_c1_wp + oc * 20, *w1 = w0 + 20, *w2 = w1 + 20, *w3 = w2 + 20;
                int32_t a0 = kws_c1_b[oc], a1 = kws_c1_b[oc + 1], a2 = kws_c1_b[oc + 2], a3 = kws_c1_b[oc + 3];
                for (int j = 0; j < 20; j++) {
                    const uint32_t xj = xv[j];
                    a0 = __SMLAD(xj, w0[j], a0);
                    a1 = __SMLAD(xj, w1[j], a1);
                    a2 = __SMLAD(xj, w2[j], a2);
                    a3 = __SMLAD(xj, w3[j], a3);
                }
                o[oc]     = sat8_relu(requant(a0, kws_c1_m[oc],     kws_c1_s[oc]),     KWS_C1_OUT_ZP);
                o[oc + 1] = sat8_relu(requant(a1, kws_c1_m[oc + 1], kws_c1_s[oc + 1]), KWS_C1_OUT_ZP);
                o[oc + 2] = sat8_relu(requant(a2, kws_c1_m[oc + 2], kws_c1_s[oc + 2]), KWS_C1_OUT_ZP);
                o[oc + 3] = sat8_relu(requant(a3, kws_c1_m[oc + 3], kws_c1_s[oc + 3]), KWS_C1_OUT_ZP);
            }
        }
    }
}

/* ---- depthwise 3x3, stride 1, SAME, 64 channels (NHWC) ---------------------------------- */
static void dwconv(const int8_t *in, int8_t *out, const int8_t *w, const int32_t *bias,
                   const int32_t *mult, const int8_t *shift, int32_t in_zp, int32_t out_zp)
{
    for (int y = 0; y < H; y++) {
        const int ky_lo = y == 0 ? 1 : 0;
        const int ky_hi = y == H - 1 ? 2 : 3;
        for (int x = 0; x < W; x++) {
            const int kx_lo = x == 0 ? 1 : 0;
            const int kx_hi = x == W - 1 ? 2 : 3;
            int32_t acc[C];
            for (int c = 0; c < C; c++) acc[c] = bias[c];
            for (int ky = ky_lo; ky < ky_hi; ky++) {
                for (int kx = kx_lo; kx < kx_hi; kx++) {
                    const int8_t *p = in + ((y + ky - 1) * W + (x + kx - 1)) * C;
                    const int8_t *q = w + (ky * 3 + kx) * C;
                    for (int c = 0; c < C; c++) acc[c] += ((int32_t)p[c] - in_zp) * q[c];
                }
            }
            int8_t *o = out + (y * W + x) * C;
            for (int c = 0; c < C; c++) o[c] = sat8_relu(requant(acc[c], mult[c], shift[c]), out_zp);
        }
    }
}

/* ---- pointwise 1x1, 64 -> 64 (input zero point already folded into bias) ----------------- */
/* Per position the 64 inputs are expanded once into 32 words of int16 lane pairs (x2[]); the
   weights are stored the same way (kws_pwN_wp, 32 words per filter).  4 filters per pass share
   each x2[] load, and every SMLAD does two multiply-accumulates. */
static void pwconv(const int8_t *in, int8_t *out, const uint32_t *w, const int32_t *bias,
                   const int32_t *mult, const int8_t *shift, int32_t out_zp)
{
    for (int p = 0; p < NPOS; p++) {
        const uint32_t *xp = (const uint32_t *)(const void *)(in + p * C);
        uint32_t x2[32];
        for (int i = 0; i < 16; i++) {
            const uint32_t a = xp[i];
            x2[2 * i]     = __SXTB16(a);                 /* (b0, b2) */
            x2[2 * i + 1] = __SXTB16(__ROR(a, 8));       /* (b1, b3) */
        }
        int8_t *o = out + p * C;
        for (int oc = 0; oc < C; oc += 4) {
            const uint32_t *w0 = w + oc * 32, *w1 = w0 + 32, *w2 = w1 + 32, *w3 = w2 + 32;
            int32_t a0 = bias[oc], a1 = bias[oc + 1], a2 = bias[oc + 2], a3 = bias[oc + 3];
            for (int i = 0; i < 32; i += 2) {
                const uint32_t xa = x2[i], xb = x2[i + 1];
                a0 = __SMLAD(xa, w0[i], a0); a0 = __SMLAD(xb, w0[i + 1], a0);
                a1 = __SMLAD(xa, w1[i], a1); a1 = __SMLAD(xb, w1[i + 1], a1);
                a2 = __SMLAD(xa, w2[i], a2); a2 = __SMLAD(xb, w2[i + 1], a2);
                a3 = __SMLAD(xa, w3[i], a3); a3 = __SMLAD(xb, w3[i + 1], a3);
            }
            o[oc]     = sat8_relu(requant(a0, mult[oc],     shift[oc]),     out_zp);
            o[oc + 1] = sat8_relu(requant(a1, mult[oc + 1], shift[oc + 1]), out_zp);
            o[oc + 2] = sat8_relu(requant(a2, mult[oc + 2], shift[oc + 2]), out_zp);
            o[oc + 3] = sat8_relu(requant(a3, mult[oc + 3], shift[oc + 3]), out_zp);
        }
    }
}

/* ---- average pool 25x5 -> 1x1, then fully connected 64 -> 12 ----------------------------- */
static void pool_fc(const int8_t *in, int8_t *logits)
{
    int8_t pooled[C];
    for (int c = 0; c < C; c++) {
        int32_t s = 0;
        for (int p = 0; p < NPOS; p++) s += in[p * C + c];
        s = s > 0 ? (s + NPOS / 2) / NPOS : (s - NPOS / 2) / NPOS;   /* TFLite int8 average pool */
        if (s < -128) s = -128;
        if (s > 127)  s = 127;
        pooled[c] = (int8_t)s;
    }
    for (int k = 0; k < KWS_NN_NUM_CLASSES; k++) {
        const int8_t *w = kws_fc_w + k * C;
        int32_t acc = kws_fc_b[k];                                   /* zero point folded in */
        for (int c = 0; c < C; c++) acc += (int32_t)pooled[c] * w[c];
        int32_t v = requant(acc, kws_fc_m[k], kws_fc_s[k]) + KWS_FC_OUT_ZP;
        if (v < -128) v = -128;
        if (v > 127)  v = 127;
        logits[k] = (int8_t)v;
    }
}

static void softmax_q(const int8_t *logits, int8_t *prob)
{
    float x[KWS_NN_NUM_CLASSES], mx = -1e30f, sum = 0.0f;
    for (int k = 0; k < KWS_NN_NUM_CLASSES; k++) {
        x[k] = (float)(logits[k] - KWS_NN_SOFTMAX_IN_ZP) * KWS_NN_SOFTMAX_IN_SCALE;
        if (x[k] > mx) mx = x[k];
    }
    for (int k = 0; k < KWS_NN_NUM_CLASSES; k++) { x[k] = expf(x[k] - mx); sum += x[k]; }
    const float inv = 256.0f / sum;
    for (int k = 0; k < KWS_NN_NUM_CLASSES; k++) {
        int32_t q = (int32_t)(x[k] * inv + 0.5f) - 128;              /* scale 1/256, zp -128 */
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        prob[k] = (int8_t)q;
    }
}

void kws_nn_run_logits(const int8_t *in, int8_t *prob, int8_t *logits)
{
    int8_t lg[KWS_NN_NUM_CLASSES];
    conv1(in, s_bufA);                                  DBG(0, s_bufA, 8000);
    dwconv(s_bufA, s_bufB, kws_dw1_w, kws_dw1_b, kws_dw1_m, kws_dw1_s, KWS_DW1_IN_ZP, KWS_DW1_OUT_ZP);   DBG(1, s_bufB, 8000);
    pwconv(s_bufB, s_bufA, kws_pw1_wp, kws_pw1_b, kws_pw1_m, kws_pw1_s, KWS_PW1_OUT_ZP);                  DBG(2, s_bufA, 8000);
    dwconv(s_bufA, s_bufB, kws_dw2_w, kws_dw2_b, kws_dw2_m, kws_dw2_s, KWS_DW2_IN_ZP, KWS_DW2_OUT_ZP);   DBG(3, s_bufB, 8000);
    pwconv(s_bufB, s_bufA, kws_pw2_wp, kws_pw2_b, kws_pw2_m, kws_pw2_s, KWS_PW2_OUT_ZP);                  DBG(4, s_bufA, 8000);
    dwconv(s_bufA, s_bufB, kws_dw3_w, kws_dw3_b, kws_dw3_m, kws_dw3_s, KWS_DW3_IN_ZP, KWS_DW3_OUT_ZP);   DBG(5, s_bufB, 8000);
    pwconv(s_bufB, s_bufA, kws_pw3_wp, kws_pw3_b, kws_pw3_m, kws_pw3_s, KWS_PW3_OUT_ZP);                  DBG(6, s_bufA, 8000);
    dwconv(s_bufA, s_bufB, kws_dw4_w, kws_dw4_b, kws_dw4_m, kws_dw4_s, KWS_DW4_IN_ZP, KWS_DW4_OUT_ZP);   DBG(7, s_bufB, 8000);
    pwconv(s_bufB, s_bufA, kws_pw4_wp, kws_pw4_b, kws_pw4_m, kws_pw4_s, KWS_PW4_OUT_ZP);                  DBG(8, s_bufA, 8000);
    pool_fc(s_bufA, lg);                                DBG(9, lg, KWS_NN_NUM_CLASSES);
    softmax_q(lg, prob);                                DBG(10, prob, KWS_NN_NUM_CLASSES);
    if (logits) for (int k = 0; k < KWS_NN_NUM_CLASSES; k++) logits[k] = lg[k];
}

void kws_nn_run(const int8_t *in, int8_t *prob)
{
    kws_nn_run_logits(in, prob, 0);
}
