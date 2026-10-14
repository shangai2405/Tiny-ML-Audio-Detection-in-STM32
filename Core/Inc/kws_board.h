/**
 * kws_board.h - board bring-up for the KWS firmware (STM32F401CC, HSI only, no crystal needed).
 *
 * Everything the project needs that CubeMX would normally generate lives in kws_board.c, so the
 * .ioc file can stay as it is and re-generating code can never wipe it out.
 *
 *   Clocks  : HSI 16 MHz -> PLL -> SYSCLK/HCLK 84 MHz, APB1 42 MHz, APB2 84 MHz
 *             PLLI2S (also from HSI) -> 128 MHz I2S clock -> exactly 16 kHz audio (div 62, odd 1)
 *   I2S2    : PB12 WS, PB13 SCK, PB15 SD  (INMP441, L/R = GND)  master RX, Philips, 32-bit frame
 *             DMA1 Stream3 / channel 0 (SPI2_RX), circular, half-word
 *   I2C1    : PB6 SCL, PB7 SDA, 400 kHz   (SSD1306 at 0x3C)
 *   USART2  : PA2 TX, PA3 RX, 9600 8N1    (HC-05 / debug)
 *   LED     : PC13 (Black Pill, active low) - configured by kws_app
 */
#ifndef KWS_BOARD_H
#define KWS_BOARD_H

#include "stm32f4xx_hal.h"

extern I2S_HandleTypeDef  hi2s2;
extern I2C_HandleTypeDef  hi2c1;
extern UART_HandleTypeDef huart2;
extern DMA_HandleTypeDef  hdma_spi2_rx;

/** Re-clock to 84 MHz and initialise I2S2 + DMA, I2C1 and USART2. Call once, before kws_app_init(). */
void kws_board_init(void);

/** Blink the on-board LED `code` times in a loop forever (used if a peripheral fails to start). */
void kws_board_fatal(int code);

#endif /* KWS_BOARD_H */
