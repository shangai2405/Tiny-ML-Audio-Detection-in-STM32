/**
 * ssd1306.h - Lightweight SSD1306 128x64 I2C OLED driver.
 *
 * Minimal footprint: 1024-byte framebuffer, tiny 5x7 and 8x16 font, no malloc.
 * Compatible with STM32 HAL I2C (blocking or DMA).
 */
#ifndef SSD1306_H
#define SSD1306_H

#include <stdint.h>
#include <stdbool.h>
#include "stm32f4xx_hal.h"

#define SSD1306_I2C_ADDR        (0x3C << 1)   /* 0x78 */
#define SSD1306_WIDTH           128
#define SSD1306_HEIGHT          64
#define SSD1306_BUFFER_SIZE     (SSD1306_WIDTH * SSD1306_HEIGHT / 8) /* 1024 bytes */

typedef enum {
    SSD1306_COLOR_BLACK = 0x00,
    SSD1306_COLOR_WHITE = 0x01
} ssd1306_color_t;

/** Initialise SSD1306 controller via I2C */
bool ssd1306_init(I2C_HandleTypeDef *hi2c);

/** Clear internal buffer */
void ssd1306_clear(void);

/** Push the 1024-byte framebuffer to OLED via I2C */
void ssd1306_update(I2C_HandleTypeDef *hi2c);

/** Draw a single pixel */
void ssd1306_draw_pixel(int16_t x, int16_t y, ssd1306_color_t color);

/** Draw a string with small 5x7 font */
void ssd1306_draw_string_small(int16_t x, int16_t y, const char *str, ssd1306_color_t color);

/** Draw a string with large font (5x7 scaled 2x) */
void ssd1306_draw_string_large(int16_t x, int16_t y, const char *str, ssd1306_color_t color);

/** Draw a string with the 5x7 font scaled by an integer factor (char pitch = 6*scale px) */
void ssd1306_draw_string_scaled(int16_t x, int16_t y, const char *str, uint8_t scale, ssd1306_color_t color);

/** Draw an outlined bar filled to pct (0..100) */
void ssd1306_draw_bar(int16_t x, int16_t y, int16_t w, int16_t h, uint8_t pct);

#endif /* SSD1306_H */
