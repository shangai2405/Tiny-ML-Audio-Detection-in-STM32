/**
 * kws_app.h - keyword spotting application: "ON" / "OFF" / other words = "UNKNOWN".
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
#define KWS_PROB_THRESHOLD_PCT  80    /* softmax threshold (%) to accept a word            */
#define KWS_CONSECUTIVE_HITS    2     /* same decision this many inferences in a row        */
#define KWS_INFER_EVERY_HOPS    15    /* 15 x 20 ms = one inference every 300 ms            */
#define KWS_REFRACTORY_MS       1000  /* ignore new decisions for 1 s after one             */
#define KWS_RESULT_HOLD_MS      2500  /* show the result this long, then back to "Listening" */

void kws_app_init(I2S_HandleTypeDef *hi2s, I2C_HandleTypeDef *hi2c, UART_HandleTypeDef *huart);
void kws_app_process(void);

#endif /* KWS_APP_H */
