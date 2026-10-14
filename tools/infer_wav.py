"""
infer_wav.py - Desktop test of the EXPORTED int8 model on WAV files, using the
exact firmware pipeline (int16 -> MFCC -> int8 quantise -> ds_cnn_s_int8.tflite
-> 4-class decision with the same threshold as the STM32).

Usage:  python3 infer_wav.py file1.wav [file2.wav ...]
        python3 infer_wav.py            (all WAVs in ../test_wavs)
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import kws_mfcc as M

ROOT = Path(__file__).resolve().parent.parent
P = json.loads((ROOT / "model" / "model_params.json").read_text())
LABELS = P["labels"]
THRESH = 0.80


def main():
    wavs = sys.argv[1:] or sorted(glob.glob(str(ROOT / "test_wavs" / "*.wav")))
    it = tf.lite.Interpreter(model_path=str(ROOT / "model" / "ds_cnn_s_int8.tflite"))
    it.allocate_tensors()
    ii, oo = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    print(f"{'file':34s} {'top-1':>10s} {'p':>5s}   p(on)  p(off)  -> board shows")
    for w in wavs:
        q = M.quantize(M.wav_to_mfcc(w).reshape(1, -1), P["input_scale"], P["input_zero_point"])
        it.set_tensor(ii, q); it.invoke()
        p = (it.get_tensor(oo)[0].astype(np.float64) - P["output_zero_point"]) * P["output_scale"]
        k = int(p.argmax())
        on, off = p[LABELS.index("on")], p[LABELS.index("off")]
        shown = "ON" if (LABELS[k] == "on" and on >= THRESH) else \
                "OFF" if (LABELS[k] == "off" and off >= THRESH) else "Listening..."
        print(f"{Path(w).name[:34]:34s} {LABELS[k]:>10s} {p[k]:5.2f}   {on:5.2f}  {off:5.2f}  -> {shown}")


if __name__ == "__main__":
    main()
