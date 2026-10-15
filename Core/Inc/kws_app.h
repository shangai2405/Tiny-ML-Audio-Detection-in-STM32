/**
 * kws_app.h - keyword spotting application: "GO" (LED on) / "STOP" (LED off) / other words = "UNKNOWN".
 *
 * Usage in main.c (already wired in this project):
 *     kws_board_init();                         // clocks + I2S/DMA, I2C, UART
 *     kws_app_init(&hi2s2, &hi2c1, &huart2);    // before while(1)
 *     kws_app_process();                        // inside while(1)
 *
 * The HAL I2S DMA callbacks live in kws_app.c - do NOT define them again in main.c.
 */
#ifndef KWS_APP_H
#define KWS_APP_H

#include "stm32f4xx_hal.h"

/* ---- tunables -------------------------------------------------------- */
/* INMP441 is quiet: the 24-bit sample is shifted right by (8 - GAIN_SHIFT) and saturated
   to int16.  4 = +24 dB.  Watch "peak=" on UART: speech at ~30 cm should peak around
   3000..20000.  Lower the value if it hits 32767 often, raise it (max 8) if it is below ~1000. */
#define KWS_MIC_GAIN_SHIFT      4
#define KWS_PROB_THRESHOLD_PCT  80    /* softmax (%) for an instant GO/STOP decision, and for UNKNOWN */
#define KWS_CMD_WEAK_PCT        50    /* GO / STOP: minimum probability (%) of the winning command            */
#define KWS_CMD_RATIO           2     /* ...and it must be this many times the other command ("stop" while "go") */
/* Voice gate: a decision is only allowed if the loudest sample of the last 1 s is at least this (16-bit scale).
   Stops GO/STOP firing on silence / room noise. Speech at ~30 cm peaks around 3000..20000; if real commands are
   ignored lower it, if noise still triggers raise it. Watch "peak=" on the UART line. */
#define KWS_VOICE_MIN_PEAK      1200
#define KWS_VOICE_HOPS          50    /* 50 x 20 ms = 1 s */
#define KWS_CONSECUTIVE_HITS    2     /* same decision this many inferences in a row        */
#define KWS_INFER_EVERY_HOPS    15    /* 15 x 20 ms = one inference every 300 ms            */
#define KWS_REFRACTORY_MS       1000  /* ignore new decisions for 1 s after one             */
/* PA0 push-to-talk (Black Pill KEY): the microphone is on only while the button is held */
#define KWS_BUTTON_ACTIVE_LOW   1     /* 1: pressed = PA0 pulled to GND (WeAct Black Pill KEY)  0: pressed = 3V3 */
#define KWS_BUTTON_DEBOUNCE_MS  30
#define KWS_PRESS_SETTLE_MS     150   /* audio ignored this long after a press (button click, pop) */
#define KWS_RELEASE_TAIL_MS     300   /* keep listening this long after release so the last word is still decided */
#define KWS_MIN_FRAMES_BEFORE_INFER 25 /* first decision after 25 x 20 ms = 0.5 s of audio (49 = a full second) */
#define KWS_RESULT_HOLD_MS      2000  /* show the result this long, then back to "Listening" */

void kws_app_init(I2S_HandleTypeDef *hi2s, I2C_HandleTypeDef *hi2c, UART_HandleTypeDef *huart);
void kws_app_process(void);

#endif /* KWS_APP_H */
