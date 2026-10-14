# KWS_F401 – "ON" / "OFF" / UNKNOWN keyword spotter (STM32F401 Black Pill)

Pretrained **ARM ML-KWS-for-MCU DS-CNN-S** (int8) running fully on the F401 with a self-contained C engine.
No X-CUBE-AI, no CMSIS-NN, no malloc. The OLED shows ON / OFF / UNKNOWN; LED PC13 follows ON/OFF.

## Wiring
| Part | Pins |
|---|---|
| INMP441 | WS=PB12, SCK=PB13, SD=PB15, L/R=GND, VDD=3V3 |
| SSD1306 (I2C 0x3C) | SCL=PB6, SDA=PB7 |
| HC-05 (USART2, 9600) | TX=PA2, RX=PA3 |

## Build / flash (STM32CubeIDE)
1. File > Import > Existing Projects into Workspace > select this folder.
2. Build (Debug is -Os). CubeIDE must compile with `-ffunction-sections -fdata-sections` and link with `--gc-sections`
   (its defaults). Without section GC the CMSIS-DSP tables overflow flash.
3. Flash. Expected size: ~85 KB flash, ~48 KB RAM.

The clock is **84 MHz** (HSI -> PLL), not 16 MHz: inference at 16 MHz would take ~0.85 s. `kws_board.c` sets this up; the .ioc is not
used by the firmware (CubeMX regeneration is safe, but keep the USER CODE lines in main.c).

## Using it
Say "on" / "off" clearly ~20-50 cm from the mic. Debug line over UART every 2 s:
`on=.. off=.. unk=.. sil=.. peak=.. ch=.. inf=..ms ovr=.. err=..`
- `inf` inference time (expect ~160 ms), `ovr` count of missed windows (should stay 0 or tiny)
- `peak` mic level: if always tiny, raise `KWS_MIC_GAIN_SHIFT`; if ~full-scale, lower it
- `err` I2S errors. Boot line prints CPU MHz and the real sample rate (should be 16000).

Tunables are in `Core/Inc/kws_app.h` (gain shift, probability threshold, consecutive hits, inference interval, hold time).

## Model tooling
- `python3 tools/export_nn.py` regenerates `Core/Src/kws_nn_data.c` from `model/ds_cnn_s_int8.tflite`.
- `python3 tools/host_test.py` checks every layer of the C engine against LiteRT (needs `ai-edge-litert`, gcc).
- `tools/infer_wav.py` runs a wav through the same MFCC + model on a PC.

## Known limits
Engine is bit-exact with TFLite on all conv layers; final FC differs by 1 LSB on ~0.2% of values.
Reference accuracy of this model on Speech Commands v2 (4590 clips): 89% 12-class; ON 86% / OFF 88% recall at p>=0.8, ~0.3% false trigger.
Real-world accuracy with an INMP441 will vary with room noise and distance.
