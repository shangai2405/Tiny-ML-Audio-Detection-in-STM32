#!/usr/bin/env python3
"""
export_nn.py - turn model/ds_cnn_s_int8.tflite (ARM ML-KWS-for-MCU DS-CNN-S, int8 PTQ)
into plain C tables for Core/Src/kws_nn_data.c, so the STM32 needs NO X-CUBE-AI.

Usage:  python3 tools/export_nn.py [path/to/model.tflite]   (default: model/ds_cnn_s_int8.tflite)
        If you pass a model (e.g. one trained with tools/train/kws_train.py) it is copied to
        model/ds_cnn_s_int8.tflite and the input/output quantisation constants in
        Core/Inc/kws_model_params.h and model/model_params.json are updated to match it.
Needs:  pip install ai-edge-litert numpy

The script asserts the exact op sequence / shapes the C engine (kws_nn.c) implements,
so it fails loudly if the .tflite is swapped for a different architecture.
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "model" / "ds_cnn_s_int8.tflite"
OUT = ROOT / "Core" / "Src" / "kws_nn_data.c"
PARAMS_H = ROOT / "Core" / "Inc" / "kws_model_params.h"
PARAMS_JSON = ROOT / "model" / "model_params.json"
if len(sys.argv) > 1:
    SRC = Path(sys.argv[1]).resolve()
    if SRC != MODEL.resolve():
        shutil.copyfile(SRC, MODEL)


def quantize_multiplier(real):
    """TFLite QuantizeMultiplier: real = q / 2^31 * 2^shift."""
    if real == 0.0:
        return 0, 0
    sig, exp = math.frexp(real)
    q = int(round(sig * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exp += 1
    if exp < -31:
        return 0, 0
    return q, exp


it = Interpreter(model_path=str(MODEL))
it.allocate_tensors()
T = {t["index"]: t for t in it.get_tensor_details()}
# RESHAPE ops only change the tensor view (49x10 -> 49x10x1 at the input, 1x1x64 -> 64 before the FC);
# the C engine's data layout is identical, so they are ignored.
ops = [o for o in it._get_ops_details() if o["op_name"] not in ("DELEGATE", "RESHAPE")]
names = [o["op_name"] for o in ops]
expect = (["CONV_2D"] + ["DEPTHWISE_CONV_2D", "CONV_2D"] * 4 +
          ["AVERAGE_POOL_2D", "FULLY_CONNECTED", "SOFTMAX"])
assert names == expect, f"unexpected graph: {names}"
assert list(T[ops[9]["inputs"][0]]["shape"]) == [1, 25, 5, 64], "avg-pool input must be 1x25x5x64"


def tensor(i):
    return it.get_tensor(i)


def qp(i):
    q = T[i]["quantization_parameters"]
    return np.array(q["scales"], dtype=np.float64), np.array(q["zero_points"], dtype=np.int64)


def layer(op, kind):
    """returns dict with weights, bias, mult, shift, zps for CONV/DW/FC op."""
    inp, w, b = op["inputs"][0], op["inputs"][1], op["inputs"][2]
    out = op["outputs"][0]
    s_in, zp_in = qp(inp)
    s_w, _ = qp(w)
    s_out, zp_out = qp(out)
    W = tensor(w).astype(np.int8)
    B = tensor(b).astype(np.int64)
    n = W.shape[0] if kind != "dw" else W.shape[3]
    if len(s_w) == 1:                       # some converters quantise the FC weights per tensor
        s_w = np.repeat(s_w, n)
    assert len(s_w) == n and np.all(_ == 0), "weights must be symmetric (zero point 0)"
    mult, shift = zip(*[quantize_multiplier(float(s_in[0] * s_w[c] / s_out[0])) for c in range(n)])
    return dict(W=W, B=B, mult=np.array(mult, np.int64), shift=np.array(shift, np.int64),
                zp_in=int(zp_in[0]), zp_out=int(zp_out[0]), n=n)


conv1 = layer(ops[0], "conv")
assert conv1["W"].shape == (64, 10, 4, 1)
dws, pws = [], []
for k in range(4):
    dw = layer(ops[1 + 2 * k], "dw")
    pw = layer(ops[2 + 2 * k], "conv")
    assert dw["W"].shape == (1, 3, 3, 64) and pw["W"].shape == (64, 1, 1, 64)
    dws.append(dw)
    pws.append(pw)
fc = layer(ops[10], "fc")
assert fc["W"].shape == (12, 64)
sm_in_scale = float(qp(ops[11]["inputs"][0])[0][0])
sm_in_zp = int(qp(ops[11]["inputs"][0])[1][0])
pool_zp = int(qp(ops[9]["outputs"][0])[1][0])
assert pool_zp == fc["zp_in"] and pool_zp == pws[3]["zp_out"]

lines = []
A = lines.append


def arr(ctype, name, data, per_line=16, align=None):
    flat = [int(v) for v in np.asarray(data).reshape(-1)]
    attr = f" __attribute__((aligned({align})))" if align else ""
    A(f"const {ctype} {name}[{len(flat)}]{attr} = {{")
    for i in range(0, len(flat), per_line):
        A("    " + ", ".join(str(v) for v in flat[i:i + per_line]) + ",")
    A("};\n")


def arr32(name, data, per_line=8):
    flat = [int(v) & 0xFFFFFFFF for v in np.asarray(data).reshape(-1)]
    A(f"const uint32_t {name}[{len(flat)}] __attribute__((aligned(4))) = {{")
    for i in range(0, len(flat), per_line):
        A("    " + ", ".join(f"0x{v:08x}u" for v in flat[i:i + per_line]) + ",")
    A("};\n")


def pack_pair(lo, hi):
    """two int8 weights -> one uint32 holding two sign-extended int16 lanes (what SXTB16 gives for activations)"""
    return (int(lo) & 0xFFFF) | ((int(hi) & 0xFFFF) << 16)


def pack_conv1(W):          # W: [64][10][4][1]  ->  [64][20] words, lane pairs = (kx0,kx1),(kx2,kx3) of each kernel row
    out = np.zeros((64, 20), dtype=np.uint64)
    for oc in range(64):
        for ky in range(10):
            for kp in range(2):
                out[oc, ky * 2 + kp] = pack_pair(W[oc, ky, 2 * kp, 0], W[oc, ky, 2 * kp + 1, 0])
    return out


def pack_pw(W):             # W: [64 oc][64 ic] -> [64][32] words: per group of 4 ic: (ic0,ic2) then (ic1,ic3)
    out = np.zeros((64, 32), dtype=np.uint64)
    for oc in range(64):
        for i in range(16):
            out[oc, 2 * i] = pack_pair(W[oc, 4 * i], W[oc, 4 * i + 2])
            out[oc, 2 * i + 1] = pack_pair(W[oc, 4 * i + 1], W[oc, 4 * i + 3])
    return out


A("/* AUTO-GENERATED by tools/export_nn.py from model/ds_cnn_s_int8.tflite - DO NOT EDIT */")
A("#include \"kws_nn.h\"\n")
A(f"const int32_t KWS_NN_SOFTMAX_IN_ZP = {sm_in_zp};")
A(f"const float   KWS_NN_SOFTMAX_IN_SCALE = {sm_in_scale!r}f;")
A(f"const int32_t KWS_NN_POOL_ZP = {pool_zp};\n")

# conv_1: weights packed as int16 lane pairs [oc][ky][kx/2] (10x4 kernel), bias NOT folded (SAME padding makes the zero-point offset position-dependent)
arr32("kws_c1_wp", pack_conv1(conv1["W"]))
arr("int32_t", "kws_c1_b", conv1["B"])
arr("int32_t", "kws_c1_m", conv1["mult"])
arr("int8_t", "kws_c1_s", conv1["shift"])
A(f"const int32_t KWS_C1_IN_ZP = {conv1['zp_in']}, KWS_C1_OUT_ZP = {conv1['zp_out']};\n")

for k in range(4):
    dw, pw = dws[k], pws[k]
    # depthwise: weights [ky][kx][c]
    arr("int8_t", f"kws_dw{k+1}_w", dw["W"], 18, 4)
    arr("int32_t", f"kws_dw{k+1}_b", dw["B"])
    arr("int32_t", f"kws_dw{k+1}_m", dw["mult"])
    arr("int8_t", f"kws_dw{k+1}_s", dw["shift"])
    A(f"const int32_t KWS_DW{k+1}_IN_ZP = {dw['zp_in']}, KWS_DW{k+1}_OUT_ZP = {dw['zp_out']};\n")
    # pointwise: weights [oc][ic]; input zero-point folded into the bias: b' = b - zp_in * sum(w)
    Wp = pw["W"].reshape(64, 64)
    bfold = pw["B"] - pw["zp_in"] * Wp.astype(np.int64).sum(axis=1)
    arr32(f"kws_pw{k+1}_wp", pack_pw(Wp))
    arr("int32_t", f"kws_pw{k+1}_b", bfold)
    arr("int32_t", f"kws_pw{k+1}_m", pw["mult"])
    arr("int8_t", f"kws_pw{k+1}_s", pw["shift"])
    A(f"const int32_t KWS_PW{k+1}_OUT_ZP = {pw['zp_out']};\n")

bf = fc["B"] - fc["zp_in"] * fc["W"].astype(np.int64).sum(axis=1)
arr("int8_t", "kws_fc_w", fc["W"], 32, 4)
arr("int32_t", "kws_fc_b", bf)
arr("int32_t", "kws_fc_m", fc["mult"])
arr("int8_t", "kws_fc_s", fc["shift"])
A(f"const int32_t KWS_FC_OUT_ZP = {fc['zp_out']};")

for name in ("mult", "shift"):
    pass
OUT.write_text("\n".join(lines) + "\n")

n_w = conv1["W"].size + sum(d["W"].size + p["W"].size for d, p in zip(dws, pws)) + fc["W"].size
print(f"wrote {OUT}  ({OUT.stat().st_size} bytes of source, {n_w} int8 weights)")

# ---- keep the firmware's input/output quantisation constants in sync with the model ----
in_q = it.get_input_details()[0]["quantization"]
out_q = it.get_output_details()[0]["quantization"]
assert abs(out_q[0] - 1 / 256) < 1e-9 and out_q[1] == -128, f"softmax output must be 1/256, zp -128 (got {out_q})"
assert list(it.get_input_details()[0]["shape"]) == [1, 490]
h = PARAMS_H.read_text()
h2 = re.sub(r"(#define\s+KWS_IN_SCALE\s+)\(.*?\)", lambda m: f"{m.group(1)}({in_q[0]:.9e}f)", h)
h2 = re.sub(r"(#define\s+KWS_IN_ZERO_POINT\s+)\(.*?\)", lambda m: f"{m.group(1)}({int(in_q[1])})", h2)
if h2 != h:
    PARAMS_H.write_text(h2)
pj = json.loads(PARAMS_JSON.read_text())
pj["input_scale"], pj["input_zero_point"] = float(in_q[0]), int(in_q[1])
PARAMS_JSON.write_text(json.dumps(pj, indent=1))
print(f"input quantisation: scale={in_q[0]:.9g} zp={int(in_q[1])}  (written to kws_model_params.h / model_params.json)")
