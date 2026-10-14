#!/usr/bin/env python3
"""
host_test.py - PC regression test for the STM32 C inference engine.

Builds Core/Src/kws_nn.c (+ kws_nn_data.c) as a shared library with the host SIMD emulation,
then feeds real MFCC inputs through BOTH the C engine and the reference TFLite interpreter
(no XNNPACK) and compares every layer output.

Usage:  python3 tools/host_test.py [more.wav ...]     (default: tools/test_wavs/*.wav)
Needs:  gcc, numpy, ai-edge-litert
"""
import ctypes, glob, json, subprocess, sys
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import kws_mfcc as M                                   # reference MFCC (bit-exact with the C one)

so = ROOT / "tools" / "_kws_nn_host.so"
subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-DKWS_NN_HOST", "-I", str(ROOT / "Core" / "Inc"),
                       str(ROOT / "Core" / "Src" / "kws_nn.c"), str(ROOT / "Core" / "Src" / "kws_nn_data.c"),
                       "-lm", "-o", str(so)])
lib = ctypes.CDLL(str(so))
dbg = (ctypes.c_int8 * (11 * 8000)).in_dll(lib, "kws_nn_dbg")
P = json.loads((ROOT / "model" / "model_params.json").read_text())

it = Interpreter(model_path=str(ROOT / "model" / "ds_cnn_s_int8.tflite"),
                 experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                 experimental_preserve_all_tensors=True)
it.allocate_tensors()
inp = it.get_input_details()[0]["index"]
LAYER_T = [23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34]     # tensor index of each C debug layer
NAMES = ["conv1", "dw1", "pw1", "dw2", "pw2", "dw3", "pw3", "dw4", "pw4", "fc logits", "probs"]
SIZES = [8000] * 9 + [12, 12]

wavs = sys.argv[1:] or sorted(glob.glob(str(ROOT / "tools" / "test_wavs" / "*.wav")))
worst = {n: 0 for n in NAMES}
mism = {n: 0 for n in NAMES}
for w in wavs:
    q = M.quantize(M.wav_to_mfcc(w).reshape(1, -1), P["input_scale"], P["input_zero_point"]).astype(np.int8)
    it.set_tensor(inp, q); it.invoke()
    prob = (ctypes.c_int8 * 12)(); logit = (ctypes.c_int8 * 12)()
    lib.kws_nn_run_logits((ctypes.c_int8 * 490)(*q.reshape(-1).tolist()), prob, logit)
    d = np.frombuffer(dbg, dtype=np.int8).reshape(11, 8000)
    for i, (n, t, sz) in enumerate(zip(NAMES, LAYER_T, SIZES)):
        ref = it.get_tensor(t).reshape(-1)
        diff = np.abs(d[i, :sz].astype(int) - ref.astype(int))
        worst[n] = max(worst[n], int(diff.max()))
        mism[n] += int((diff != 0).sum())
for n in NAMES:
    print(f"{n:10s} max|diff| = {worst[n]}   mismatching values = {mism[n]}")
print(f"{len(wavs)} clips")
