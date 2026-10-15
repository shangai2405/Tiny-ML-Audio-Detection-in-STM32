/**
 * kws_app.c - keyword spotting for STM32F401CC + INMP441 + SSD1306 (+ HC-05 debug link).
 *
 * Push-to-talk: the microphone is ON only while the PA0 button (onboard KEY) is held down (plus a short
 * KWS_RELEASE_TAIL_MS so the word you just said is still decided); the I2S clock keeps running but the audio is discarded otherwise.
 *
 * Data flow
 *   I2S2 (master RX, Philips, 32-bit frame, 16 kHz) --circular DMA--> s_dma[]
 *   DMA half/full ISR: pick mic channel, 24->16 bit, push into s_ring[] (ISR is ~10 us)
 *   main loop: every 320 new samples (20 ms) -> 640-sample window -> MFCC (CMSIS-DSP)
 *              -> int8 -> rolling 49x10 feature matrix (= the last 1 s of audio)
 *   every KWS_INFER_EVERY_HOPS hops: DS-CNN-S (kws_nn.c, no X-CUBE-AI)
 *              -> threshold + consecutive hits -> GO / STOP / UNKNOWN -> LED, OLED, UART
 *
 * Decision rule: top-1 class must be above KWS_PROB_THRESHOLD_PCT, KWS_CONSECUTIVE_HITS times
 * in a row.  "go" -> GO (LED on), "stop" -> STOP (LED off), any other word class
 * (on/off/yes/no/up/down/left/right or the model's own _unknown_ class) -> UNKNOWN.  Silence never triggers.
 * (go and stop are the only commands; everything else is shown as UNKNOWN.)
 * The confidence shown on the OLED is frozen at the highest value reached while the word is held.
 *
 * RAM: dma 2.5 KB + ring 16 KB + window 1.25 KB + features 0.5 KB + FFT 8 KB + OLED 1 KB
 *      + NN activations 16 KB + UART queue 0.5 KB  => ~46 KB of 64 KB.   No malloc, no float printf.
 */
#include "kws_app.h"
#include "kws_mfcc.h"
#include "kws_model_params.h"
#include "kws_nn.h"
#include "ssd1306.h"
#include <stdio.h>
#include <string.h>

#ifndef KWS_IDX_GO
#define KWS_IDX_GO    11
#endif
#ifndef KWS_IDX_STOP
#define KWS_IDX_STOP  10
#endif

/* ---------------------------------------------------------------- audio buffers */
#define DMA_HALF_SAMPLES   160                        /* 10 ms per DMA half            */
#define DMA_HALFWORDS      (DMA_HALF_SAMPLES * 4 * 2) /* L(2)+R(2) halfwords, 2 halves   */
#define RING_SAMPLES       8192                       /* 512 ms of slack, power of 2     */

static uint16_t s_dma[DMA_HALFWORDS];
static int16_t  s_ring[RING_SAMPLES];
static volatile uint32_t s_ring_wr;       /* total samples written (ISR)   */
static uint32_t s_ring_rd;                /* total samples consumed (main) */
static volatile uint32_t s_overruns;
static volatile uint8_t  s_i2s_error;     /* set by HAL_I2S_ErrorCallback, handled in main loop */
static uint32_t s_i2s_restarts;

/* mic channel auto-detect: INMP441 L/R=GND -> left slot; the other slot is tri-stated */
static volatile int      s_chan = -1;     /* -1 = detecting, 0 = left, 1 = right */
static volatile uint32_t s_energy[2];
static volatile uint32_t s_detect_halves;
static volatile int32_t  s_peak;

/* ---------------------------------------------------------------- features / model */
static int16_t s_win[KWS_WIN_LEN];
static int8_t  s_feat[KWS_FEATURE_SIZE];
static uint32_t s_frames_total;
static uint32_t s_hops_since_infer;
static int8_t  s_out_q[KWS_NUM_CLASSES];
static uint8_t s_pct[KWS_NUM_CLASSES];    /* class probabilities in % */
static uint32_t s_last_ms_infer;
static uint16_t s_hop_peak[KWS_VOICE_HOPS];   /* loudest sample of each of the last 1 s of 20 ms hops (voice gate) */
static uint8_t  s_hop_idx;

/* ---------------------------------------------------------------- app state */
typedef enum { ST_LISTEN = 0, ST_GO, ST_STOP, ST_UNKNOWN } kws_state_t;
enum { CAT_NONE = -1, CAT_GO = 0, CAT_STOP = 1, CAT_UNK = 2 };

static kws_state_t s_state = ST_LISTEN;
static int8_t   s_cat_last = CAT_NONE;
static uint8_t  s_cat_hits;
static uint8_t  s_shown_idx;              /* class index shown on the OLED / UART */
static uint8_t  s_disp_pct;               /* confidence shown on the OLED: peak held while the word is shown */
static uint32_t s_last_trigger_ms;
static uint32_t s_last_oled_ms;
static uint8_t  s_oled_ok, s_oled_dirty = 1;

/* ---------------------------------------------------------------- PA0 push button / microphone on-off */
static uint8_t  s_mic_on;                 /* 1 while the I2S microphone is running */
static uint8_t  s_btn_stable, s_btn_raw;  /* debounced / raw "button is held" */
static uint32_t s_btn_change_ms;
static uint32_t s_settle_until;           /* audio is discarded until this tick after a press */
static uint32_t s_release_ms;             /* when the button was released (mic stops KWS_RELEASE_TAIL_MS later) */

static I2S_HandleTypeDef  *s_i2s;
static I2C_HandleTypeDef  *s_i2c;
static UART_HandleTypeDef *s_uart;

/* ---------------------------------------------------------------- UART (interrupt driven TX queue) */
#define TX_RING 512
static uint8_t  s_tx[TX_RING];
static volatile uint16_t s_tx_head, s_tx_tail, s_tx_len;
static volatile uint8_t  s_tx_busy;

static void uart_kick(void)          /* call with interrupts masked, or from the UART ISR */
{
    if (s_tx_busy || s_tx_head == s_tx_tail) return;
    uint16_t n = (s_tx_head > s_tx_tail) ? (uint16_t)(s_tx_head - s_tx_tail) : (uint16_t)(TX_RING - s_tx_tail);
    s_tx_len = n;
    s_tx_busy = 1;
    if (HAL_UART_Transmit_IT(s_uart, &s_tx[s_tx_tail], n) != HAL_OK) s_tx_busy = 0;
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart != s_uart) return;
    s_tx_tail = (uint16_t)((s_tx_tail + s_tx_len) % TX_RING);
    s_tx_busy = 0;
    uart_kick();
}

static void uart_puts(const char *s)
{
    if (!s_uart) return;
    uint32_t pm = __get_PRIMASK();
    __disable_irq();
    while (*s) {
        uint16_t nx = (uint16_t)((s_tx_head + 1) % TX_RING);
        if (nx == s_tx_tail) break;                 /* queue full: drop the rest (debug text only) */
        s_tx[s_tx_head] = (uint8_t)*s++;
        s_tx_head = nx;
    }
    uart_kick();
    __set_PRIMASK(pm);
}

/* ---------------------------------------------------------------- helpers */
static void led_set(int on)   /* Black Pill LED on PC13, active LOW */
{
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, on ? GPIO_PIN_RESET : GPIO_PIN_SET);
}

static void led_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOC_CLK_ENABLE();
    g.Pin = GPIO_PIN_13; g.Mode = GPIO_MODE_OUTPUT_PP; g.Pull = GPIO_NOPULL; g.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &g);
    led_set(0);
}

static void cycle_counter_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* INMP441: 24-bit MSB-first in a 32-bit slot. DMA stores 2 halfwords per slot:
   [0] = bits 31..16 (sample bits 23..8), [1] = bits 15..0 (sample bits 7..0 + 8 zero bits) */
static inline int16_t slot_to_pcm16(const uint16_t *hw)
{
    int32_t s24 = ((int32_t)(int16_t)hw[0] << 8) | (hw[1] >> 8);  /* sign-extended 24-bit */
    int32_t v = s24 >> (8 - KWS_MIC_GAIN_SHIFT);
    if (v > 32767)  v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

/* ---------------------------------------------------------------- DMA ISR side */
static void push_half(const uint16_t *half)
{
    if (s_chan < 0) {                       /* first ~0.5 s: find which slot has the mic */
        uint32_t e0 = 0, e1 = 0;
        for (int i = 0; i < DMA_HALF_SAMPLES; i++) {
            int16_t l = slot_to_pcm16(&half[4 * i]), r = slot_to_pcm16(&half[4 * i + 2]);
            e0 += (uint32_t)(l < 0 ? -l : l);
            e1 += (uint32_t)(r < 0 ? -r : r);
        }
        s_energy[0] += e0 >> 4; s_energy[1] += e1 >> 4;
        if (++s_detect_halves >= 50) s_chan = (s_energy[1] > s_energy[0]) ? 1 : 0;
        return;
    }
    uint32_t wr = s_ring_wr;
    int32_t peak = s_peak;
    const uint16_t *p = &half[2 * s_chan];
    for (int i = 0; i < DMA_HALF_SAMPLES; i++, p += 4) {
        int16_t v = slot_to_pcm16(p);
        s_ring[(wr + i) & (RING_SAMPLES - 1)] = v;
        int32_t a = v < 0 ? -v : v;
        if (a > peak) peak = a;
    }
    s_peak = peak;
    s_ring_wr = wr + DMA_HALF_SAMPLES;
}

/* strong definitions override the HAL weak ones - do NOT also define them in main.c */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s == s_i2s) { push_half(&s_dma[0]); }
}

void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s == s_i2s) { push_half(&s_dma[DMA_HALFWORDS / 2]); }
}

void HAL_I2S_ErrorCallback(I2S_HandleTypeDef *hi2s)
{
    if (hi2s == s_i2s) s_i2s_error = 1;     /* restarted from the main loop */
}

/* ---------------------------------------------------------------- PA0 button (polled, debounced) */
static void button_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOA_CLK_ENABLE();
    g.Pin   = GPIO_PIN_0;
    g.Mode  = GPIO_MODE_INPUT;
#if KWS_BUTTON_ACTIVE_LOW
    g.Pull  = GPIO_PULLUP;                /* Black Pill KEY: PA0 to GND when pressed */
#else
    g.Pull  = GPIO_PULLDOWN;
#endif
    HAL_GPIO_Init(GPIOA, &g);
}

static int button_held_now(void)
{
    int level = (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_SET);
#if KWS_BUTTON_ACTIVE_LOW
    return !level;
#else
    return level;
#endif
}

/* returns the debounced state: 1 = held */
static uint8_t button_update(uint32_t now)
{
    uint8_t raw = (uint8_t)button_held_now();
    if (raw != s_btn_raw) { s_btn_raw = raw; s_btn_change_ms = now; }
    else if (raw != s_btn_stable && (now - s_btn_change_ms) >= KWS_BUTTON_DEBOUNCE_MS) s_btn_stable = raw;
    return s_btn_stable;
}

/* ---------------------------------------------------------------- OLED */
static void oled_draw(void)
{
    char line[28];
    if (!s_oled_ok) return;
    ssd1306_clear();
    if (!s_mic_on && s_state == ST_LISTEN) {
        ssd1306_draw_string_scaled(1, 6, "MIC OFF", 3, SSD1306_COLOR_WHITE);
        ssd1306_draw_string_small(4, 38, "hold the PA0 button", SSD1306_COLOR_WHITE);
        ssd1306_draw_string_small(4, 50, "and say a word", SSD1306_COLOR_WHITE);
    } else if (s_state == ST_GO || s_state == ST_STOP) {
        const char *w = (s_state == ST_GO) ? "GO" : "STOP";
        int16_t x = (int16_t)((SSD1306_WIDTH - (int)strlen(w) * 6 * 4) / 2);
        ssd1306_draw_string_scaled(x, 4, w, 4, SSD1306_COLOR_WHITE);
        snprintf(line, sizeof(line), "conf %u%%", s_disp_pct);
        ssd1306_draw_string_small(4, 40, line, SSD1306_COLOR_WHITE);
        ssd1306_draw_bar(4, 52, 120, 10, s_disp_pct);
    } else if (s_state == ST_UNKNOWN) {
        ssd1306_draw_string_scaled(1, 6, "UNKNOWN", 3, SSD1306_COLOR_WHITE);       /* 7 chars x 18 px = 126 px */
        if (s_shown_idx == KWS_IDX_UNKNOWN)
            snprintf(line, sizeof(line), "other word %u%%", s_disp_pct);
        else
            snprintf(line, sizeof(line), "heard: %s %u%%", KWS_LABELS[s_shown_idx], s_disp_pct);
        ssd1306_draw_string_small(4, 40, line, SSD1306_COLOR_WHITE);
        ssd1306_draw_bar(4, 52, 120, 10, s_disp_pct);
    } else {
        /* idle screen: one live probability bar for each command */
        ssd1306_draw_string_scaled(2, 9, "go", 2, SSD1306_COLOR_WHITE);
        ssd1306_draw_bar(54, 6, 72, 18, s_pct[KWS_IDX_GO]);
        ssd1306_draw_string_scaled(2, 41, "stop", 2, SSD1306_COLOR_WHITE);
        ssd1306_draw_bar(54, 38, 72, 18, s_pct[KWS_IDX_STOP]);
    }
    ssd1306_update(s_i2c);          /* ~25 ms blocking; audio keeps flowing into s_ring */
}

/* ---------------------------------------------------------------- init */
static void i2s_start(void)
{
    /* Size argument = number of 32-bit data items; HAL doubles it for 32-bit frames. */
    if (HAL_I2S_Receive_DMA(s_i2s, s_dma, DMA_HALFWORDS / 2) != HAL_OK)
        uart_puts("I2S DMA start FAILED\r\n");
}

static void mic_start(void)
{
    uint32_t now = HAL_GetTick();
    s_ring_rd = s_ring_wr;                    /* drop anything old: the 1 s feature window starts empty */
    s_frames_total = 0;
    s_hops_since_infer = 0;
    memset(s_win, 0, sizeof(s_win));
    memset(s_feat, (int8_t)KWS_IN_ZERO_POINT, sizeof(s_feat));
    memset(s_pct, 0, sizeof(s_pct));
    memset(s_hop_peak, 0, sizeof(s_hop_peak));
    s_cat_last = CAT_NONE; s_cat_hits = 0;
    s_peak = 0;
    s_state = ST_LISTEN;
    s_last_trigger_ms = now - KWS_REFRACTORY_MS;
    s_i2s_error = 0;
    s_settle_until = now + KWS_PRESS_SETTLE_MS;   /* ignore button click / mic pop right after the press */
    s_mic_on = 1;
    s_oled_dirty = 1;
    uart_puts("MIC ON\r\n");
}

static void mic_stop(void)
{
    /* the I2S clock keeps running (no INMP441 start-up pop on the next press); audio is simply ignored */
    s_mic_on = 0;                             /* a result that is still on screen stays until its hold time ends */
    s_oled_dirty = 1;
    uart_puts("MIC OFF\r\n");
}

void kws_app_init(I2S_HandleTypeDef *hi2s, I2C_HandleTypeDef *hi2c, UART_HandleTypeDef *huart)
{
    char msg[96];
    s_i2s = hi2s; s_i2c = hi2c; s_uart = huart;

    led_init();
    button_init();
    cycle_counter_init();
    uart_puts("\r\n=== STM32F401 KWS (go/stop) - hold PA0 to listen ===\r\n");

    s_oled_ok = ssd1306_init(s_i2c) ? 1 : 0;
    uart_puts(s_oled_ok ? "OLED ok\r\n" : "OLED not found at 0x3C (check PB6/PB7)\r\n");
    if (s_oled_ok) {
        ssd1306_clear();
        ssd1306_draw_string_large(4, 20, "KWS boot", SSD1306_COLOR_WHITE);
        ssd1306_update(s_i2c);
    }

    if (kws_mfcc_init() != 0) uart_puts("MFCC init FAILED\r\n");

    snprintf(msg, sizeof(msg), "CPU %lu MHz, NN ready (ds_cnn_s int8)\r\n", (unsigned long)(SystemCoreClock / 1000000U));
    uart_puts(msg);

    memset(s_win, 0, sizeof(s_win));
    memset(s_feat, (int8_t)KWS_IN_ZERO_POINT, sizeof(s_feat));
    i2s_start();                              /* clocked continuously; only *used* while PA0 is held */

    /* the microphone stays off until the PA0 button is clicked */
    /* Actual audio rate: HAL derives it from the I2S prescaler it programmed. */
    {
        uint32_t i2scfgr = hi2s->Instance->I2SPR;
        uint32_t div = i2scfgr & 0xFFu, odd = (i2scfgr >> 8) & 1u;
        uint32_t i2sclk = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_I2S);
        uint32_t fs = (div * 2u + odd) ? i2sclk / (64u * (div * 2u + odd)) : 0;
        snprintf(msg, sizeof(msg), "I2S2 ready (mic off, hold PA0), Fs=%lu Hz (div=%lu odd=%lu)\r\n",
                 (unsigned long)fs, (unsigned long)div, (unsigned long)odd);
        uart_puts(msg);
    }
}

/* ---------------------------------------------------------------- inference + decision */
static void run_inference(void)
{
    char msg[96];

    uint32_t t0 = DWT->CYCCNT;
    kws_nn_run(s_feat, s_out_q);
    s_last_ms_infer = (DWT->CYCCNT - t0) / (SystemCoreClock / 1000U);

    for (int i = 0; i < KWS_NUM_CLASSES; i++) {
        int32_t v = ((int32_t)s_out_q[i] - KWS_OUT_ZERO_POINT) * 100 / 256;  /* scale = 1/256 */
        s_pct[i] = (uint8_t)(v < 0 ? 0 : v > 100 ? 100 : v);
    }

    int top = 0;
    for (int i = 1; i < KWS_NUM_CLASSES; i++) if (s_pct[i] > s_pct[top]) top = i;

    /* category of a confident top-1: GO, STOP or UNKNOWN (any other word); silence never counts */
    /* GO / STOP are decided against EACH OTHER: the winner must reach KWS_CMD_WEAK_PCT and be at least
       KWS_CMD_RATIO times the other command. A clear win (>= KWS_PROB_THRESHOLD_PCT) fires at once, a
       weaker one needs KWS_CONSECUTIVE_HITS inferences in a row. A little "stop" while you say "go" no
       longer blocks the decision. */
    int cat = CAT_NONE, strong = 0;
    int pg = s_pct[KWS_IDX_GO], ps = s_pct[KWS_IDX_STOP];
    if (top == KWS_IDX_GO && pg >= KWS_CMD_WEAK_PCT && pg >= KWS_CMD_RATIO * ps) {
        cat = CAT_GO;   strong = (pg >= KWS_PROB_THRESHOLD_PCT);
    } else if (top == KWS_IDX_STOP && ps >= KWS_CMD_WEAK_PCT && ps >= KWS_CMD_RATIO * pg) {
        cat = CAT_STOP; strong = (ps >= KWS_PROB_THRESHOLD_PCT);
    } else if (top != KWS_IDX_SILENCE && top != KWS_IDX_GO && top != KWS_IDX_STOP && s_pct[top] >= KWS_PROB_THRESHOLD_PCT) {
        cat = CAT_UNK;
    }

    /* voice gate: nothing is decided unless the last second actually contained sound above the room-noise level */
    uint16_t loud = 0;
    for (int i = 0; i < KWS_VOICE_HOPS; i++) if (s_hop_peak[i] > loud) loud = s_hop_peak[i];
    if (loud < KWS_VOICE_MIN_PEAK) cat = CAT_NONE;

    if (cat == CAT_NONE)            { s_cat_last = CAT_NONE; s_cat_hits = 0; }
    else if (cat == s_cat_last)     { if (s_cat_hits < 255) s_cat_hits++; }
    else                            { s_cat_last = (int8_t)cat; s_cat_hits = 1; }
    if (strong && cat != CAT_UNK && s_cat_hits < KWS_CONSECUTIVE_HITS) s_cat_hits = KWS_CONSECUTIVE_HITS;

    uint32_t now = HAL_GetTick();
    int refractory = (now - s_last_trigger_ms) < KWS_REFRACTORY_MS;
    if (!refractory && s_cat_hits >= KWS_CONSECUTIVE_HITS) {
        s_shown_idx = (uint8_t)top;
        s_disp_pct  = s_pct[top];
        s_state = (cat == CAT_GO) ? ST_GO : (cat == CAT_STOP) ? ST_STOP : ST_UNKNOWN;
        if      (cat == CAT_GO)   led_set(1);               /* GO   : LED on  */
        else if (cat == CAT_STOP) led_set(0);               /* STOP : LED off */
        /* UNKNOWN leaves the LED as it is */
        s_last_trigger_ms = now;
        s_cat_last = CAT_NONE; s_cat_hits = 0;
        s_oled_dirty = 1;
        if (cat == CAT_UNK)
            snprintf(msg, sizeof(msg), ">>> UNKNOWN (%s %u%%)\r\n", KWS_LABELS[top], s_pct[top]);
        else
            snprintf(msg, sizeof(msg), ">>> %s (%u%%)\r\n",
                     cat == CAT_GO ? "GO" : "STOP", s_pct[top]);
        uart_puts(msg);
    } else if (s_state != ST_LISTEN) {
        /* while the result is shown, keep the highest confidence of the same word and freeze it there */
        if (top == s_shown_idx && s_pct[top] > s_disp_pct) {
            s_disp_pct = s_pct[top];
            s_oled_dirty = 1;
        }
        if ((now - s_last_trigger_ms) > KWS_RESULT_HOLD_MS) {
            snprintf(msg, sizeof(msg), "peak %s %u%%\r\n", KWS_LABELS[s_shown_idx], s_disp_pct);
            uart_puts(msg);
            s_state = ST_LISTEN;          /* LED keeps its last on/off state */
            s_oled_dirty = 1;
        }
    }
}

/* ---------------------------------------------------------------- main loop */
void kws_app_process(void)
{
    static uint32_t last_dbg_ms;
    char msg[160];

    uint32_t now0 = HAL_GetTick();

    /* recover from an I2S error (overrun etc.): restart the DMA stream (both states) */
    if (s_i2s_error) {
        s_i2s_error = 0;
        s_i2s_restarts++;
        HAL_I2S_DMAStop(s_i2s);
        i2s_start();
        s_ring_rd = s_ring_wr;
    }

    /* PA0 held = microphone used; released = ignored after a short tail */
    uint8_t held = button_update(now0);
    if (held && !s_mic_on) mic_start();
    if (held) s_release_ms = now0;
    if (s_mic_on && !held && (now0 - s_release_ms) >= KWS_RELEASE_TAIL_MS) mic_stop();
    if (!s_mic_on) {
        s_ring_rd = s_ring_wr;                /* throw the audio away */
        /* idle: expire a shown result */
        if (s_state != ST_LISTEN && (now0 - s_last_trigger_ms) > KWS_RESULT_HOLD_MS) {
            char m2[48];
            snprintf(m2, sizeof(m2), "peak %s %u%%\r\n", KWS_LABELS[s_shown_idx], s_disp_pct);
            uart_puts(m2);
            s_state = ST_LISTEN;
            s_oled_dirty = 1;
        }
        if (s_oled_dirty) { s_oled_dirty = 0; s_last_oled_ms = now0; oled_draw(); }
        return;
    }

    /* 1) consume all complete 20 ms hops from the ring buffer */
    for (;;) {
        uint32_t wr = s_ring_wr;
        if ((int32_t)(now0 - s_settle_until) < 0) { s_ring_rd = wr; break; }   /* press settle: discard */
        if (wr - s_ring_rd > RING_SAMPLES - DMA_HALF_SAMPLES) {     /* fell behind: resync */
            s_overruns++;
            s_ring_rd = wr - KWS_HOP_LEN;
        }
        if (wr - s_ring_rd < KWS_HOP_LEN) break;

        memmove(&s_win[0], &s_win[KWS_HOP_LEN], (KWS_WIN_LEN - KWS_HOP_LEN) * sizeof(int16_t));
        for (int i = 0; i < KWS_HOP_LEN; i++)
            s_win[KWS_WIN_LEN - KWS_HOP_LEN + i] = s_ring[(s_ring_rd + i) & (RING_SAMPLES - 1)];
        s_ring_rd += KWS_HOP_LEN;

        {   /* loudness of this hop for the voice gate */
            int32_t hp = 0;
            for (int i = KWS_WIN_LEN - KWS_HOP_LEN; i < KWS_WIN_LEN; i++) {
                int32_t a = s_win[i] < 0 ? -s_win[i] : s_win[i];
                if (a > hp) hp = a;
            }
            s_hop_peak[s_hop_idx] = (uint16_t)(hp > 65535 ? 65535 : hp);
            if (++s_hop_idx >= KWS_VOICE_HOPS) s_hop_idx = 0;
        }
        float mf[KWS_NUM_MFCC];
        kws_mfcc_frame(s_win, mf);
        memmove(&s_feat[0], &s_feat[KWS_NUM_MFCC], (KWS_NUM_FRAMES - 1) * KWS_NUM_MFCC);
        kws_mfcc_quantize(mf, &s_feat[(KWS_NUM_FRAMES - 1) * KWS_NUM_MFCC], KWS_NUM_MFCC,
                          KWS_IN_SCALE, KWS_IN_ZERO_POINT);
        if (s_frames_total == 0) {          /* push-to-talk start: pad the window with the first (room-noise) frame */
            const int8_t *first = &s_feat[(KWS_NUM_FRAMES - 1) * KWS_NUM_MFCC];
            for (int r = 0; r < KWS_NUM_FRAMES - 1; r++) memcpy(&s_feat[r * KWS_NUM_MFCC], first, KWS_NUM_MFCC);
        }
        s_frames_total++;
        s_hops_since_infer++;
    }

    /* 2) inference every KWS_INFER_EVERY_HOPS hops once a full 1 s window exists */
    if (s_frames_total >= KWS_MIN_FRAMES_BEFORE_INFER && s_hops_since_infer >= KWS_INFER_EVERY_HOPS) {
        s_hops_since_infer = 0;
        run_inference();
    }

    uint32_t now = HAL_GetTick();

    /* 3) OLED: on state change, otherwise every 500 ms (bars) */
    if (s_oled_dirty || (now - s_last_oled_ms) >= 500) {
        s_oled_dirty = 0;
        s_last_oled_ms = now;
        oled_draw();
    }

    /* 4) UART debug every 2 s (queued, never blocks) */
    if (now - last_dbg_ms >= 2000) {
        last_dbg_ms = now;
        int32_t pk = s_peak; s_peak = 0;
        snprintf(msg, sizeof(msg), "go=%u stop=%u unk=%u sil=%u peak=%ld ch=%d inf=%lums ovr=%lu err=%lu\r\n",
                 s_pct[KWS_IDX_GO], s_pct[KWS_IDX_STOP],
                 s_pct[KWS_IDX_UNKNOWN], s_pct[KWS_IDX_SILENCE],
                 (long)pk, s_chan, (unsigned long)s_last_ms_infer, (unsigned long)s_overruns,
                 (unsigned long)s_i2s_restarts);
        uart_puts(msg);
    }
}
