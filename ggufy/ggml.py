"""GGML type definitions and quantization/dequantization.

Faithful Python port of ggml's quantization (the `*_ref` deterministic
implementations). All hot loops are vectorised with numpy across blocks so
converting real model tensors stays practical.

Block sizes / byte-per-block come from src/Gguf.zig (GgmlType) and match the
ggml-common.h block structs.
"""

from __future__ import annotations

import math
import struct
from typing import List

import numpy as np

QK_K = 256


def fp16_to_fp32(h) -> np.float32:
    return np.float32(np.float16(h))


def fp32_to_fp16(x) -> np.uint16:
    return np.uint16(np.float16(np.float32(x)).view(np.uint16))


def fp32_to_fp16_bytes(x) -> bytes:
    return np.float16(np.float32(x)).view(np.uint16).tobytes()


def fp16_bytes_to_fp32(b) -> np.float32:
    return np.float32(np.frombuffer(b, dtype=np.float16)[0])


def _fp16_pack(arr: np.ndarray) -> np.ndarray:
    """Pack a float32 array of scale values into (N,2) uint8 little-endian."""
    h = np.asarray(arr, dtype=np.float32).astype(np.float16).view(np.uint16)
    return np.stack([h & 0xFF, h >> 8], axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# GGML type registry (from src/Gguf.zig GgmlType)
# ---------------------------------------------------------------------------

GGML_TYPE_IDS = {
    "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q4_2": 4, "q4_3": 5,
    "q5_0": 6, "q5_1": 7, "q8_0": 8, "q8_1": 9, "q2_k": 10, "q3_k": 11,
    "q4_k": 12, "q5_k": 13, "q6_k": 14, "q8_k": 15, "iq2_xxs": 16,
    "iq2_xs": 17, "iq3_xxs": 18, "iq1_s": 19, "iq4_nl": 20, "iq3_s": 21,
    "iq2_s": 22, "iq4_xs": 23, "i8": 24, "i16": 25, "i32": 26, "i64": 27,
    "f64": 28, "iq1_m": 29, "bf16": 30, "q4_0_4_4": 31, "q4_0_4_8": 32,
    "q4_0_8_8": 33, "tq1_0": 34, "tq2_0": 35, "iq4_nl_4_4": 36,
    "iq4_nl_4_8": 37, "iq4_nl_8_8": 38, "mxfp4": 39, "nvfp4": 40,
    "q1_0": 41,
}

UNSUPPORTED_TYPES = {
    "q4_2", "q4_3", "q4_0_4_4", "q4_0_4_8", "q4_0_8_8",
    "iq4_nl_4_4", "iq4_nl_4_8", "iq4_nl_8_8",
}

BLOCK_SIZES = {
    "q4_0": 32, "q4_1": 32, "q5_0": 32, "q5_1": 32, "q8_0": 32,
    "q8_1": 32, "q2_k": 256, "q3_k": 256, "q4_k": 256, "q5_k": 256,
    "q6_k": 256, "q8_k": 256, "iq2_xxs": 256, "iq2_xs": 256,
    "iq3_xxs": 256, "iq1_s": 256, "iq4_nl": 256, "iq3_s": 256,
    "iq2_s": 256, "iq4_xs": 256, "iq1_m": 256, "mxfp4": 32, "nvfp4": 64,
}

BYTES_PER_BLOCK = {
    "f32": 4, "f16": 2, "bf16": 2, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f64": 8,
    "q4_0": 18, "q4_1": 20, "q5_0": 22, "q5_1": 24, "q8_0": 34, "q8_1": 36,
    "q2_k": 84, "q3_k": 110, "q4_k": 144, "q5_k": 176, "q6_k": 210,
    "q8_k": 292, "mxfp4": 17, "nvfp4": 36,
}


class GgmlType:
    @staticmethod
    def from_string(value: str) -> str:
        lower = value.lower()
        if lower not in GGML_TYPE_IDS:
            raise ValueError(f"InvalidGgmlType: {value}")
        return lower

    @staticmethod
    def from_int(value: int) -> str:
        for name, i in GGML_TYPE_IDS.items():
            if i == value:
                return name
        raise ValueError(f"InvalidGgmlType: {value}")

    @staticmethod
    def from_safetensors_type(s: str) -> str:
        m = {
            "F32": "f32", "F16": "f16", "BF16": "bf16", "I32": "i32",
            "I16": "i16", "I8": "i8", "F64": "f64", "I64": "i64",
        }
        if s in m:
            return m[s]
        raise ValueError(f"InvalidGgmlType: {s}")

    @staticmethod
    def is_unsupported(t: str) -> bool:
        return t in UNSUPPORTED_TYPES

    @staticmethod
    def get_block_size(t: str) -> int:
        return BLOCK_SIZES.get(t, 1)

    @staticmethod
    def get_bytes_per_block(t: str) -> int:
        return BYTES_PER_BLOCK.get(t, 0)

    @staticmethod
    def calc_size_in_bytes(t: str, n_elements: int) -> int:
        if t in ("f32", "i32"):
            return 4 * n_elements
        if t in ("f16", "bf16", "i16"):
            return 2 * n_elements
        if t in ("f64", "i64"):
            return 8 * n_elements
        if t == "i8":
            return n_elements
        bs = GgmlType.get_block_size(t)
        return (n_elements // bs) * GgmlType.get_bytes_per_block(t)


# ---------------------------------------------------------------------------
# e2m1 (FP4) values doubled, shared by MXFP4/NVFP4 (from ggml-common.h)
# ---------------------------------------------------------------------------
KVALUES_FP4 = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int8,
)


def e8m0_to_fp32_half(e: int) -> np.float32:
    if e == 0:
        return np.float32(np.float32(2.0 ** -126) * np.float32(0.5))
    return np.float32(2.0 ** np.float32(e - 128))


def e8m0_to_fp32(e: int) -> np.float32:
    if e == 0:
        return np.float32(2.0 ** -127)
    if e == 255:
        return np.float32(np.nan)
    return np.float32(2.0 ** (e - 127))


# ---------------------------------------------------------------------------
# Generic block-wise quantize entry point (like ggml_quantize_chunk)
# ---------------------------------------------------------------------------

def quantize_chunk(q_type: str, x: np.ndarray, start: int, nrows: int,
                   n_per_row: int) -> bytes:
    slice_ = x[start * n_per_row:(start + nrows) * n_per_row]
    return quantize_row(q_type, slice_)


def quantize_row(q_type: str, x: np.ndarray) -> bytes:
    x = np.ascontiguousarray(x, dtype=np.float32)
    fn = {
        "q4_0": quantize_row_q4_0,
        "q4_1": quantize_row_q4_1,
        "q5_0": quantize_row_q5_0,
        "q5_1": quantize_row_q5_1,
        "q8_0": quantize_row_q8_0,
        "q2_k": quantize_row_q2_k,
        "q3_k": quantize_row_q3_k,
        "q4_k": quantize_row_q4_k,
        "q5_k": quantize_row_q5_k,
        "q6_k": quantize_row_q6_k,
        "mxfp4": quantize_row_mxfp4,
    }
    if q_type not in fn:
        raise ValueError(f"UnsupportedDestinationType: {q_type}")
    return fn[q_type](x)


def dequantize_row(q_type: str, data: bytes, n_elements: int) -> np.ndarray:
    fn = {
        "f32": dequantize_row_f32,
        "f16": dequantize_row_f16,
        "bf16": dequantize_row_bf16,
        "f64": dequantize_row_f64,
        "i8": dequantize_row_i8,
        "i16": dequantize_row_i16,
        "i32": dequantize_row_i32,
        "i64": dequantize_row_i64,
        "q4_0": dequantize_row_q4_0,
        "q4_1": dequantize_row_q4_1,
        "q5_0": dequantize_row_q5_0,
        "q5_1": dequantize_row_q5_1,
        "q8_0": dequantize_row_q8_0,
        "q8_1": dequantize_row_q8_1,
        "q2_k": dequantize_row_q2_k,
        "q3_k": dequantize_row_q3_k,
        "q4_k": dequantize_row_q4_k,
        "q5_k": dequantize_row_q5_k,
        "q6_k": dequantize_row_q6_k,
        "mxfp4": dequantize_row_mxfp4,
    }
    if q_type not in fn:
        raise ValueError(f"UnsupportedSourceType: {q_type}")
    return fn[q_type](data, n_elements)


# ---------------------------------------------------------------------------
# Scalar float helpers
# ---------------------------------------------------------------------------

def roundf(x) -> float:
    """C roundf: round half away from zero."""
    f = float(x)
    if f >= 0:
        return math.floor(f + 0.5)
    return math.ceil(f - 0.5)


def nearest_int(fval) -> int:
    """ggml nearest_int: round-to-nearest-even on the f32 value."""
    return int(round(float(np.float32(fval))))


# ---------------------------------------------------------------------------
# 4/5/8-bit block quantization (vectorised over blocks of 32)
# ---------------------------------------------------------------------------

def quantize_row_q4_0(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    absv = np.abs(xb)
    jmax = np.argmax(absv, axis=1)
    rows = np.arange(nb)
    amax = absv[rows, jmax]
    maxv = xb[rows, jmax]
    d = np.where(amax > 0, maxv / np.float32(-8.0), np.float32(0.0))
    id_ = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    scaled = xb * id_[:, None] + np.float32(8.5)
    xi = np.minimum(np.float32(15.0), np.floor(scaled)).astype(np.uint8)
    lo = xi[:, 0:16]
    hi = xi[:, 16:32]
    qs = (lo & 0x0F) | ((hi & 0x0F) << 4)
    out = np.empty((nb, 18), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(d)
    out[:, 2:18] = qs
    return out.tobytes()


def dequantize_row_q4_0(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 18).reshape(nb, 18)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qs = arr[:, 2:18]
    lo = ((qs & 0x0F).astype(np.int32) - 8).astype(np.float32)
    hi = ((qs >> 4).astype(np.int32) - 8).astype(np.float32)
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(n)


def quantize_row_q4_1(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    mn = np.min(xb, axis=1)
    mx = np.max(xb, axis=1)
    d = (mx - mn) / np.float32(15.0)
    id_ = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    scaled = (xb - mn[:, None]) * id_[:, None] + np.float32(0.5)
    xi = np.minimum(np.float32(15.0), np.floor(scaled)).astype(np.uint8)
    lo = xi[:, 0:16]
    hi = xi[:, 16:32]
    qs = (lo & 0x0F) | ((hi & 0x0F) << 4)
    out = np.empty((nb, 20), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(d)
    out[:, 2:4] = _fp16_pack(mn)
    out[:, 4:20] = qs
    return out.tobytes()


def dequantize_row_q4_1(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 20).reshape(nb, 20)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    m = np.ascontiguousarray(arr[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qs = arr[:, 4:20]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None] + m[:, None]
    out[:, 16:32] = hi * d[:, None] + m[:, None]
    return out.reshape(n)


def quantize_row_q5_0(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    absv = np.abs(xb)
    jmax = np.argmax(absv, axis=1)
    rows = np.arange(nb)
    amax = absv[rows, jmax]
    maxv = xb[rows, jmax]
    d = np.where(amax > 0, maxv / np.float32(-16.0), np.float32(0.0))
    id_ = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    scaled = xb * id_[:, None] + np.float32(16.5)
    xi = np.minimum(np.float32(31.0), np.floor(scaled)).astype(np.uint32)
    lo = xi[:, 0:16]
    hi = xi[:, 16:32]
    qs = ((lo & 0x0F) | ((hi & 0x0F) << 4)).astype(np.uint8)
    # qh: bit j = high bit of element j (j 0..15), bit j+16 = high bit of j+16
    bits_lo = ((lo & 0x10) >> 4).astype(np.uint32)
    bits_hi = ((hi & 0x10) >> 4).astype(np.uint32)
    jbits = (np.uint32(1) << np.arange(16)).astype(np.uint32)
    qh32 = np.sum(bits_lo * jbits[None, :], axis=1) | \
           np.sum(bits_hi * (jbits << np.uint32(16))[None, :], axis=1)
    qh_bytes = qh32.astype(np.uint32)
    qh_le = np.empty((nb, 4), dtype=np.uint8)
    qh_le[:, 0] = qh_bytes & 0xFF
    qh_le[:, 1] = (qh_bytes >> 8) & 0xFF
    qh_le[:, 2] = (qh_bytes >> 16) & 0xFF
    qh_le[:, 3] = (qh_bytes >> 24) & 0xFF
    out = np.empty((nb, 22), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(d)
    out[:, 2:6] = qh_le
    out[:, 6:22] = qs
    return out.tobytes()


def dequantize_row_q5_0(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 22).reshape(nb, 22)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qh = np.ascontiguousarray(arr[:, 2:6]).view(np.uint32)
    qs = arr[:, 6:22]
    j = np.arange(16)
    xh_0 = (((qh >> (j + 0)) << 4) & 0x10).astype(np.uint32)
    xh_1 = ((qh >> (j + 12)) & 0x10).astype(np.uint32)
    lo = ((qs[:, :16] & 0x0F).astype(np.uint32) | xh_0).astype(np.int32) - 16
    hi = ((qs[:, :16] >> 4).astype(np.uint32) | xh_1).astype(np.int32) - 16
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(n)


def quantize_row_q5_1(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    mn = np.min(xb, axis=1)
    mx = np.max(xb, axis=1)
    d = (mx - mn) / np.float32(31.0)
    id_ = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    scaled = (xb - mn[:, None]) * id_[:, None] + np.float32(0.5)
    xi = np.floor(scaled).astype(np.uint32)
    lo = xi[:, 0:16]
    hi = xi[:, 16:32]
    qs = ((lo & 0x0F) | ((hi & 0x0F) << 4)).astype(np.uint8)
    bits_lo = ((lo & 0x10) >> 4).astype(np.uint32)
    bits_hi = ((hi & 0x10) >> 4).astype(np.uint32)
    jbits = (np.uint32(1) << np.arange(16)).astype(np.uint32)
    qh32 = np.sum(bits_lo * jbits[None, :], axis=1) | \
           np.sum(bits_hi * (jbits << np.uint32(16))[None, :], axis=1)
    qh_le = np.empty((nb, 4), dtype=np.uint8)
    qh_le[:, 0] = qh32 & 0xFF
    qh_le[:, 1] = (qh32 >> 8) & 0xFF
    qh_le[:, 2] = (qh32 >> 16) & 0xFF
    qh_le[:, 3] = (qh32 >> 24) & 0xFF
    out = np.empty((nb, 24), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(d)
    out[:, 2:4] = _fp16_pack(mn)
    out[:, 4:8] = qh_le
    out[:, 8:24] = qs
    return out.tobytes()


def dequantize_row_q5_1(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 24).reshape(nb, 24)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    m = np.ascontiguousarray(arr[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qh = np.ascontiguousarray(arr[:, 4:8]).view(np.uint32)
    qs = arr[:, 8:24]
    j = np.arange(16)
    xh_0 = (((qh >> (j + 0)) << 4) & 0x10).astype(np.uint32)
    xh_1 = ((qh >> (j + 12)) & 0x10).astype(np.uint32)
    lo = ((qs[:, :16] & 0x0F).astype(np.uint32) | xh_0).astype(np.float32)
    hi = ((qs[:, :16] >> 4).astype(np.uint32) | xh_1).astype(np.float32)
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None] + m[:, None]
    out[:, 16:32] = hi * d[:, None] + m[:, None]
    return out.reshape(n)


def quantize_row_q8_0(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    amax = np.max(np.abs(xb), axis=1)
    d = amax / np.float32(127.0)
    id_ = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    scaled = xb * id_[:, None]
    # roundf: half away from zero
    q = np.where(scaled >= 0, np.floor(scaled + np.float32(0.5)),
                 np.ceil(scaled - np.float32(0.5)))
    qs = q.astype(np.int8).astype(np.uint8)
    out = np.empty((nb, 34), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(d)
    out[:, 2:34] = qs
    return out.tobytes()


def dequantize_row_q8_0(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 34).reshape(nb, 34)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qs = np.ascontiguousarray(arr[:, 2:34]).view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(n)


def dequantize_row_q8_1(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 36).reshape(nb, 36)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    qs = np.ascontiguousarray(arr[:, 4:36]).view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(n)


def quantize_row_mxfp4(x: np.ndarray) -> bytes:
    nb = x.size // 32
    xb = x.reshape(nb, 32)
    amax = np.max(np.abs(xb), axis=1)
    e = np.zeros(nb, dtype=np.uint8)
    nz = amax > 0
    e[nz] = (np.floor(np.log2(amax[nz])).astype(np.int32) - 2 + 127).astype(np.uint8)
    d = np.empty(nb, dtype=np.float32)
    for i in range(nb):
        d[i] = e8m0_to_fp32_half(int(e[i]))
    # best_index_mxfp4 for each element: nearest kvalues*d
    vals = xb.reshape(nb, 32)  # (nb, 32)
    d2 = d.reshape(nb, 1, 1)
    k = KVALUES_FP4.astype(np.float32).reshape(1, 16, 1) * d2  # (nb,16,32)
    err = np.abs(k - vals[:, None, :])  # (nb,16,32)
    best = np.argmin(err, axis=1)  # (nb,32)
    lo = best[:, 0:16]
    hi = best[:, 16:32]
    qs = (lo | (hi << 4)).astype(np.uint8)
    out = np.empty((nb, 17), dtype=np.uint8)
    out[:, 0] = e
    out[:, 1:17] = qs
    return out.tobytes()


def dequantize_row_mxfp4(data: bytes, n: int) -> np.ndarray:
    nb = n // 32
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 17).reshape(nb, 17)
    e = arr[:, 0]
    qs = arr[:, 1:17]
    d = np.empty(nb, dtype=np.float32)
    for i in range(nb):
        d[i] = e8m0_to_fp32_half(int(e[i]))
    lo = KVALUES_FP4[qs & 0x0F].astype(np.float32)
    hi = KVALUES_FP4[qs >> 4].astype(np.float32)
    out = np.empty((nb, 32), dtype=np.float32)
    out[:, 0:16] = lo * d[:, None]
    out[:, 16:32] = hi * d[:, None]
    return out.reshape(n)


# ---------------------------------------------------------------------------
# Plain float/int types
# ---------------------------------------------------------------------------

def dequantize_row_f32(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32, count=n).copy()


def dequantize_row_f16(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float16, count=n).astype(np.float32)


def dequantize_row_bf16(data: bytes, n: int) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16, count=n)
    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def dequantize_row_f64(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float64, count=n).astype(np.float32)


def dequantize_row_i8(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int8, count=n).astype(np.float32)


def dequantize_row_i16(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16, count=n).astype(np.float32)


def dequantize_row_i32(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int32, count=n).astype(np.float32)


def dequantize_row_i64(data: bytes, n: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int64, count=n).astype(np.float32)


# ---------------------------------------------------------------------------
# Super-block (K) quantization helpers (vectorised over groups/blocks)
# ---------------------------------------------------------------------------

GROUP_MAX_EPS = 1e-15


def _nearest_int_array(vals: np.ndarray) -> np.ndarray:
    return np.rint(vals).astype(np.int64)


def make_qkx2_quants_batch(n, nmax, x, weights, rmin, rdelta, nstep, use_mad):
    """make_qkx2_quants for many groups at once.

    x, weights: (G, n) float64. Returns (scale (G,), the_min (G,), L (G, n) uint8).
    Mirrors the ggml reference (including the initial candidate and the
    nstep+1 sweep) on a per-group basis. The nstep sweep is vectorised across
    candidates; groups are processed in chunks to bound memory.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    mn = np.min(x, axis=1)
    mx = np.max(x, axis=1)
    mn = np.where(mn > 0, 0.0, mn).astype(np.float32)
    flat = (mx == mn)
    denom = np.maximum(mx - mn, np.float32(1e-30))
    iscale = nmax / denom
    scale = np.where(flat, 0.0, 1.0 / iscale).astype(np.float32)
    the_min = -mn

    G = x.shape[0]
    L = np.zeros((G, n), dtype=np.uint8)

    # Initial candidate
    l = np.clip(_nearest_int_array(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    L[:] = l
    diff = scale[:, None] * l + mn[:, None] - x
    diff = np.abs(diff) if use_mad else diff * diff
    best_error = np.sum(weights * diff, axis=1)

    if nstep < 1:
        return scale, the_min, L

    sum_w = np.sum(weights, axis=1)
    sum_x = np.sum(weights * x, axis=1)
    Laux = np.empty_like(L)

    for is_ in range(nstep + 1):
        iscale2 = (rmin + rdelta * is_ + nmax) / np.maximum(mx - mn, np.float32(1e-30))
        l2 = np.clip(_nearest_int_array(iscale2[:, None] * (x - mn[:, None])), 0, nmax)
        Laux[:] = l2
        wl = weights * l2
        sum_l = wl.sum(axis=1)
        sum_l2 = (wl * l2).sum(axis=1)
        sum_xl = (wl * x).sum(axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        this_scale = np.where(ok, (sum_w * sum_xl - sum_x * sum_l) / D, 0.0)
        this_min = np.where(ok, (sum_l2 * sum_x - sum_l * sum_xl) / D, 0.0)
        clamp = this_min > 0
        this_min = np.where(clamp, 0.0, this_min)
        this_scale = np.where(clamp, sum_xl / np.maximum(sum_l2, 1e-30), this_scale)
        diff = this_scale[:, None] * l2 + this_min[:, None] - x
        diff = np.abs(diff) if use_mad else diff * diff
        cur_error = np.sum(weights * diff, axis=1)
        better = ok & (cur_error < best_error)
        if not np.any(better):
            continue
        scale = np.where(better, this_scale, scale)
        mn = np.where(better, this_min, mn)
        best_error = np.where(better, cur_error, best_error)
        L = np.where(better[:, None], Laux, L)

    return scale, the_min, L


def get_scale_min_k4(j, scales):
    """Extract scale (d) and min (m) for group j from the packed scales array.
    scales may be a flat 12-element array (scalar path) or (N,12)."""
    scalar = scales.ndim == 1
    if j < 4:
        d = scales[..., j] & 63
        m = scales[..., j + 4] & 63
    else:
        d = (scales[..., j + 4] & 0xF) | ((scales[..., j - 4] >> 6) << 4)
        m = (scales[..., j + 4] >> 4) | ((scales[..., j] >> 6) << 4)
    return int(d) if scalar else d, (int(m) if scalar else m)


# ---------------------------------------------------------------------------
# K-quantization (vectorised over blocks)
# ---------------------------------------------------------------------------

def quantize_row_q2_k(x: np.ndarray) -> bytes:
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    G = nb * 16
    segs = xb.reshape(G, 16).astype(np.float32)
    weights = np.abs(segs)
    scales, mins, Lmat = make_qkx2_quants_batch(16, 3, segs, weights, -0.5, 0.1, 15, True)
    scales = scales.reshape(nb, 16)
    mins = mins.reshape(nb, 16)
    Lmat = Lmat.reshape(nb, 16, 16)

    q4scale = 15.0
    max_scale = np.max(scales, axis=1)
    max_min = np.max(mins, axis=1)

    out = np.zeros((nb, 84), dtype=np.uint8)
    # d / dmin
    nz = max_scale > 0
    out[:, 80:82] = _fp16_pack(np.where(nz, max_scale / q4scale, 0.0))
    out[:, 82:84] = _fp16_pack(np.where(max_min > 0, max_min / q4scale, 0.0))

    sc_lo = np.zeros((nb, 16), dtype=np.uint8)
    sc_lo[nz] = (np.rint(q4scale * scales[nz] / max_scale[nz, None]).astype(np.int64) & 0xFF).astype(np.uint8)
    sc_hi = np.zeros((nb, 16), dtype=np.uint8)
    mnz = max_min > 0
    sc_hi[mnz] = (np.rint(q4scale * mins[mnz] / max_min[mnz, None]).astype(np.int64) & 0xFF).astype(np.uint8)
    out[:, 0:16] = sc_lo | (sc_hi << 4)

    d = np.ascontiguousarray(out[:, 80:82]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    dmin = np.ascontiguousarray(out[:, 82:84]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    packed = out[:, 0:16].astype(np.uint32)
    sc = packed & 0xF
    dm = packed >> 4
    d_j = d[:, None] * sc
    dm_j = dmin[:, None] * dm
    L2 = np.rint((segs.reshape(nb, 16, 16) + dm_j[:, :, None]) / d_j[:, :, None])
    L2 = np.clip(L2, 0, 3).astype(np.uint8).reshape(nb, 16 * 16)

    qs = np.zeros((nb, 64), dtype=np.uint8)
    for jj in range(0, QK_K, 128):
        idx = jj // 4
        blk = L2[:, jj:jj + 128]
        for l in range(32):
            qs[:, idx + l] = (blk[:, l] | (blk[:, l + 32] << 2) |
                              (blk[:, l + 64] << 4) | (blk[:, l + 96] << 6)).astype(np.uint8)
    out[:, 16:80] = qs
    return out.tobytes()


def dequantize_row_q2_k(data: bytes, n: int) -> np.ndarray:
    nb = n // QK_K
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 84).reshape(nb, 84)
    scales = arr[:, 0:16].astype(np.uint32)
    q = arr[:, 16:80]
    d = np.ascontiguousarray(arr[:, 80:82]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    mn = np.ascontiguousarray(arr[:, 82:84]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for nn in range(0, QK_K, 128):
        qoff = nn // 4
        qb = q[:, qoff:qoff + 32]
        is_ = (nn // 128) * 8
        for j in range(4):
            shift = 2 * j
            sc = scales[:, is_ + 2 * j]
            dl = d * (sc & 0xF)
            ml = mn * (sc >> 4)
            out[:, nn + 32 * j:nn + 32 * j + 16] = \
                dl[:, None] * (((qb[:, 0:16] >> shift) & 3)) - ml[:, None]
            sc = scales[:, is_ + 2 * j + 1]
            dl = d * (sc & 0xF)
            ml = mn * (sc >> 4)
            out[:, nn + 32 * j + 16:nn + 32 * j + 32] = \
                dl[:, None] * (((qb[:, 16:32] >> shift) & 3)) - ml[:, None]
    return out.reshape(n)


def quantize_row_q3_k(x: np.ndarray) -> bytes:
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    # make_q3_quants per 16-group (rmse path) - vectorised outer loop, inner
    # refinement is small.
    G = nb * 16
    segs = xb.reshape(G, 16).astype(np.float32)
    Lmat, scales = _make_q3_quants_batch(16, 4, segs)
    scales = scales.reshape(nb, 16)
    Lmat = Lmat.reshape(nb, 16, 16)

    amax = np.max(np.abs(scales), axis=1)
    imax = np.argmax(np.abs(scales), axis=1)
    max_scale = scales[np.arange(nb), imax]
    nz = amax > GROUP_MAX_EPS

    out = np.zeros((nb, 110), dtype=np.uint8)
    sc_out = np.zeros((nb, 12), dtype=np.uint8)
    d_block = np.zeros(nb)
    iscale = np.full(nb, -1.0)
    iscale[nz] = -32.0 / max_scale[nz]
    d_block[nz] = 1.0 / iscale[nz]
    out[:, 108:110] = _fp16_pack(d_block)

    # Pack scales
    ls = np.rint(iscale[:, None] * scales).astype(np.int64)
    ls = np.clip(ls, -32, 31) + 32
    # distribute l into the 12 scale bytes (mirrors reference)
    for j in range(16):
        lv = ls[:, j]  # (nb,)
        low = lv & 0xF
        high = (lv >> 4) & 0x3
        if j < 8:
            sc_out[:, j] |= low.astype(np.uint8)
        else:
            sc_out[:, j - 8] |= (low.astype(np.uint8) << 4)
        sc_out[:, j % 4 + 8] |= (high.astype(np.uint8) << (2 * (j // 4)))
    out[:, 96:108] = sc_out

    # requantize
    d = np.ascontiguousarray(out[:, 108:110]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    L = np.zeros((nb, QK_K), dtype=np.int64)
    for j in range(16):
        sc = sc_out[:, j].astype(np.int32) & 0xF if j < 8 else (sc_out[:, j - 8].astype(np.int32) >> 4)
        sc = (sc | (((sc_out[:, 8 + j % 4].astype(np.int32) >> (2 * (j // 4))) & 3) << 4)) - 32
        dj = d * sc
        z = dj != 0
        seg = segs.reshape(nb, 16, 16)[:, j, :]
        l = np.rint(seg / dj[:, None])
        l = np.clip(l, -4, 3).astype(np.int64)
        L[:, 16 * j:16 * j + 16] = np.where(z[:, None], l + 4, 0)

    hmask = np.zeros((nb, 32), dtype=np.uint8)
    for j in range(QK_K):
        m = j % 32
        hm = 1 << (j // 32)
        gt = L[:, j] > 3
        hmask[:, m] |= np.where(gt, hm, 0).astype(np.uint8)
        L[:, j] = np.where(gt, L[:, j] - 4, L[:, j])
    out[:, 0:32] = hmask

    qs = np.zeros((nb, 64), dtype=np.uint8)
    for jj in range(0, QK_K, 128):
        idx = jj // 4
        blk = L[:, jj:jj + 128]
        for l in range(32):
            qs[:, idx + l] = (blk[:, l] | (blk[:, l + 32] << 2) |
                              (blk[:, l + 64] << 4) | (blk[:, l + 96] << 6)).astype(np.uint8)
    out[:, 32:96] = qs
    return out.tobytes()


def _make_q3_quants_batch(n, nmax, x):
    """make_q3_quants (do_rmse=True) for many groups at once.
    x: (G, n) float64. Returns (L (G,n) int64, scale (G,))."""
    ax = np.abs(x)
    i = np.argmax(ax, axis=1)
    rows = np.arange(x.shape[0])
    amax = ax[rows, i]
    max_ = x[rows, i]
    nz = amax >= GROUP_MAX_EPS
    iscale = np.where(nz, -nmax / np.where(max_ != 0, max_, 1.0), 0.0)
    l = _nearest_int_array(iscale[:, None] * x)
    l = np.clip(l, -nmax, nmax - 1)
    L = l.astype(np.int64)
    w = x * x
    sumlx = np.sum(w * x * l, axis=1)
    suml2 = np.sum(w * l * l, axis=1)
    for _ in range(5):
        n_changed = 0
        for i2 in range(n):
            wi = w[:, i2]
            xv = x[:, i2]
            Lv = L[:, i2]
            slx = sumlx - wi * xv * Lv
            pos = slx > 0
            if not np.any(pos):
                continue
            sl2 = suml2 - wi * Lv * Lv
            new_l = _nearest_int_array(xv * sl2 / np.where(slx != 0, slx, 1.0))
            new_l = np.clip(new_l, -nmax, nmax - 1)
            diff = new_l != Lv
            upd = pos & diff
            if not np.any(upd):
                continue
            slx2 = slx + wi * xv * new_l
            sl2_2 = sl2 + wi * new_l * new_l
            better = (sl2_2 > 0) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl2_2) & upd
            if not np.any(better):
                continue
            L[better, i2] = new_l[better]
            sumlx = np.where(better, slx2, sumlx)
            suml2 = np.where(better, sl2_2, suml2)
            n_changed += int(np.count_nonzero(better))
        if n_changed == 0:
            break
    L = L + nmax
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 != 0, suml2, 1.0), 0.0)
    scale = np.where(nz, scale, 0.0)
    return L, scale


def dequantize_row_q3_k(data: bytes, n: int) -> np.ndarray:
    nb = n // QK_K
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 110).reshape(nb, 110)
    hm = arr[:, 0:32]
    q = arr[:, 32:96]
    sc_raw = arr[:, 96:108]
    d_all = np.ascontiguousarray(arr[:, 108:110]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)

    kmask1 = 0x03030303
    kmask2 = 0x0f0f0f0f
    aux = np.zeros((nb, 4), dtype=np.uint32)
    aux[:, 0] = sc_raw[:, 0] | (sc_raw[:, 1].astype(np.uint32) << 8) | (sc_raw[:, 2].astype(np.uint32) << 16) | (sc_raw[:, 3].astype(np.uint32) << 24)
    aux[:, 1] = sc_raw[:, 4] | (sc_raw[:, 5].astype(np.uint32) << 8) | (sc_raw[:, 6].astype(np.uint32) << 16) | (sc_raw[:, 7].astype(np.uint32) << 24)
    aux[:, 2] = sc_raw[:, 8] | (sc_raw[:, 9].astype(np.uint32) << 8) | (sc_raw[:, 10].astype(np.uint32) << 16) | (sc_raw[:, 11].astype(np.uint32) << 24)
    tmp = aux[:, 2].astype(np.uint64)
    naux = np.zeros((nb, 4), dtype=np.uint32)
    naux[:, 2] = ((aux[:, 0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    naux[:, 3] = ((aux[:, 1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    naux[:, 0] = (aux[:, 0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    naux[:, 1] = (aux[:, 1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    scales = np.zeros((nb, 16), dtype=np.int8)
    for k in range(16):
        b = k // 4
        o = (k % 4) * 8
        scales[:, k] = ((naux[:, b] >> o) & 0xFF).astype(np.uint8).view(np.int8)

    out = np.empty((nb, QK_K), dtype=np.float32)
    m_arr = np.ones((nb, 1), dtype=np.uint8)
    for nn in range(0, QK_K, 128):
        qoff = nn // 4
        qb = q[:, qoff:qoff + 32]
        base_is = (nn // 128) * 8
        for j in range(4):
            dl = d_all * (scales[:, base_is + 2 * j].astype(np.float32) - 32)
            hmv = ((hm[:, 0:16] & m_arr) == 0).astype(np.int32) * 4
            v = ((qb[:, 0:16] >> (2 * j)) & 3).astype(np.int32) - hmv
            out[:, nn + 32 * j:nn + 32 * j + 16] = dl[:, None] * v
            dl = d_all * (scales[:, base_is + 2 * j + 1].astype(np.float32) - 32)
            hmv = ((hm[:, 16:32] & m_arr) == 0).astype(np.int32) * 4
            v = ((qb[:, 16:32] >> (2 * j)) & 3).astype(np.int32) - hmv
            out[:, nn + 32 * j + 16:nn + 32 * j + 32] = dl[:, None] * v
            m_arr = (m_arr << 1) & 0xFF
    return out.reshape(n)


def quantize_row_q4_k(x: np.ndarray) -> bytes:
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    G = nb * 8
    segs = xb.reshape(G, 32).astype(np.float32)
    sum_x2 = np.sum(segs * segs, axis=1)
    av_x = np.sqrt(sum_x2 / 32)
    weights = av_x[:, None] + np.abs(segs)
    scales, mins, Lmat = make_qkx2_quants_batch(32, 15, segs, weights, -1.0, 0.1, 20, False)
    scales = scales.reshape(nb, 8)
    mins = mins.reshape(nb, 8)
    Lmat = Lmat.reshape(nb, 8, 32)

    max_scale = np.max(scales, axis=1)
    max_min = np.max(mins, axis=1)
    inv_scale = np.where(max_scale > 0, 63.0 / max_scale, 0.0)
    inv_min = np.where(max_min > 0, 63.0 / max_min, 0.0)

    ls = np.clip(np.rint(inv_scale[:, None] * scales).astype(np.int64), 0, 63)
    lm = np.clip(np.rint(inv_min[:, None] * mins).astype(np.int64), 0, 63)

    out = np.zeros((nb, 144), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(np.where(max_scale > 0, max_scale / 63.0, 0.0))
    out[:, 2:4] = _fp16_pack(np.where(max_min > 0, max_min / 63.0, 0.0))

    sc_arr = np.zeros((nb, 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc_arr[:, j] = ls[:, j].astype(np.uint8)
            sc_arr[:, j + 4] = lm[:, j].astype(np.uint8)
        else:
            sc_arr[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).astype(np.uint8)
            sc_arr[:, j - 4] |= ((ls[:, j] >> 4) << 6).astype(np.uint8)
            sc_arr[:, j] |= ((lm[:, j] >> 4) << 6).astype(np.uint8)
    out[:, 4:16] = sc_arr

    d = np.ascontiguousarray(out[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    dmin = np.ascontiguousarray(out[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    sc, m = np.zeros((nb, 8), dtype=np.uint8), np.zeros((nb, 8), dtype=np.uint8)
    for j in range(8):
        sc[:, j], m[:, j] = get_scale_min_k4(j, sc_arr)
    dj = d[:, None] * sc.astype(np.float32)
    dm = dmin[:, None] * m.astype(np.float32)
    L2 = np.rint((segs.reshape(nb, 8, 32) + dm[:, :, None]) / dj[:, :, None])
    L2 = np.clip(L2, 0, 15).astype(np.uint8).reshape(nb, 8 * 32)

    qs = np.zeros((nb, 128), dtype=np.uint8)
    for jj in range(0, QK_K, 64):
        idx = jj // 2
        blk = L2[:, jj:jj + 64]
        for l in range(32):
            qs[:, idx + l] = (blk[:, l] | (blk[:, l + 32] << 4)).astype(np.uint8)
    out[:, 16:144] = qs
    return out.tobytes()


def dequantize_row_q4_k(data: bytes, n: int) -> np.ndarray:
    nb = n // QK_K
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 144).reshape(nb, 144)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    mn = np.ascontiguousarray(arr[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    scales = arr[:, 4:16].astype(np.uint8)
    q = arr[:, 16:144]
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(0, QK_K, 64):
        qoff = j // 2
        qb = q[:, qoff:qoff + 32]
        g = j // 64 * 2
        for grp in range(2):
            sc, m = get_scale_min_k4(g + grp, scales)  # (nb,)
            d1 = d * sc
            m1 = mn * m
            if grp == 0:
                out[:, j:j + 32] = d1[:, None] * (qb & 0x0F).astype(np.float32) - m1[:, None]
            else:
                out[:, j + 32:j + 64] = d1[:, None] * (qb >> 4).astype(np.float32) - m1[:, None]
    return out.reshape(n)


def quantize_row_q5_k(x: np.ndarray) -> bytes:
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    G = nb * 8
    segs = xb.reshape(G, 32).astype(np.float32)
    sum_x2 = np.sum(segs * segs, axis=1)
    av_x = np.sqrt(sum_x2 / 32)
    weights = av_x[:, None] + np.abs(segs)
    scales, mins, Lmat = make_qkx2_quants_batch(32, 31, segs, weights, -0.5, 0.1, 15, False)
    scales = scales.reshape(nb, 8)
    mins = mins.reshape(nb, 8)
    Lmat = Lmat.reshape(nb, 8, 32)

    max_scale = np.max(scales, axis=1)
    max_min = np.max(mins, axis=1)
    inv_scale = np.where(max_scale > 0, 63.0 / max_scale, 0.0)
    inv_min = np.where(max_min > 0, 63.0 / max_min, 0.0)
    ls = np.clip(np.rint(inv_scale[:, None] * scales).astype(np.int64), 0, 63)
    lm = np.clip(np.rint(inv_min[:, None] * mins).astype(np.int64), 0, 63)

    out = np.zeros((nb, 176), dtype=np.uint8)
    out[:, 0:2] = _fp16_pack(np.where(max_scale > 0, max_scale / 63.0, 0.0))
    out[:, 2:4] = _fp16_pack(np.where(max_min > 0, max_min / 63.0, 0.0))
    sc_arr = np.zeros((nb, 12), dtype=np.uint8)
    for j in range(8):
        if j < 4:
            sc_arr[:, j] = ls[:, j].astype(np.uint8)
            sc_arr[:, j + 4] = lm[:, j].astype(np.uint8)
        else:
            sc_arr[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).astype(np.uint8)
            sc_arr[:, j - 4] |= ((ls[:, j] >> 4) << 6).astype(np.uint8)
            sc_arr[:, j] |= ((lm[:, j] >> 4) << 6).astype(np.uint8)
    out[:, 4:16] = sc_arr

    d = np.ascontiguousarray(out[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    dmin = np.ascontiguousarray(out[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    sc, m = np.zeros((nb, 8), dtype=np.uint8), np.zeros((nb, 8), dtype=np.uint8)
    for j in range(8):
        sc[:, j], m[:, j] = get_scale_min_k4(j, sc_arr)
    dj = d[:, None] * sc.astype(np.float32)
    dm = dmin[:, None] * m.astype(np.float32)
    L2 = np.rint((segs.reshape(nb, 8, 32) + dm[:, :, None]) / dj[:, :, None])
    L2 = np.clip(L2, 0, 31).astype(np.uint8).reshape(nb, 8 * 32)

    qh = np.zeros((nb, 32), dtype=np.uint8)
    ql = np.zeros((nb, 128), dtype=np.uint8)
    for n_ in range(0, QK_K, 64):
        blk = L2[:, n_:n_ + 64]
        m1 = 1 << (2 * (n_ // 64))
        m2 = 2 << (2 * (n_ // 64))
        l1 = blk[:, 0:32].astype(np.int32)
        l2 = blk[:, 32:64].astype(np.int32)
        hi1 = l1 > 15
        hi2 = l2 > 15
        l1 = np.where(hi1, l1 - 16, l1)
        l2 = np.where(hi2, l2 - 16, l2)
        qh |= (hi1.astype(np.uint8) * m1) | (hi2.astype(np.uint8) * m2)
        ql[:, (n_ // 64) * 32:(n_ // 64 + 1) * 32] = (l1 | (l2 << 4)).astype(np.uint8)
    out[:, 16:48] = qh
    out[:, 48:176] = ql
    return out.tobytes()


def dequantize_row_q5_k(data: bytes, n: int) -> np.ndarray:
    nb = n // QK_K
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 176).reshape(nb, 176)
    d = np.ascontiguousarray(arr[:, 0:2]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    mn = np.ascontiguousarray(arr[:, 2:4]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    scales = arr[:, 4:16].astype(np.uint8)
    qh = arr[:, 16:48]
    ql = arr[:, 48:176]
    out = np.empty((nb, QK_K), dtype=np.float32)
    for j in range(0, QK_K, 64):
        qoff = j // 2
        qb = ql[:, qoff:qoff + 32]
        u1 = 1 << (2 * (j // 64))
        u2 = 2 << (2 * (j // 64))
        g = 2 * (j // 64)
        sc, m = get_scale_min_k4(g + 0, scales)
        d1 = d * sc
        m1 = mn * m
        v = (qb & 0x0F).astype(np.int32) + np.where((qh & u1) != 0, 16, 0)
        out[:, j:j + 32] = d1[:, None] * v - m1[:, None]
        sc, m = get_scale_min_k4(g + 1, scales)
        d2 = d * sc
        m2 = mn * m
        v = (qb >> 4).astype(np.int32) + np.where((qh & u2) != 0, 16, 0)
        out[:, j + 32:j + 64] = d2[:, None] * v - m2[:, None]
    return out.reshape(n)


def quantize_row_q6_k(x: np.ndarray) -> bytes:
    nb = x.size // QK_K
    xb = x.reshape(nb, QK_K)
    G = nb * 16
    segs = xb.reshape(G, 16).astype(np.float32)
    L, scales = _make_qx_quants_batch(16, 32, segs)
    scales = scales.reshape(nb, 16)
    L = L.reshape(nb, 16, 16)

    max_abs_scale = np.max(np.abs(scales), axis=1)
    imax = np.argmax(np.abs(scales), axis=1)
    max_scale = scales[np.arange(nb), imax]
    eps = max_abs_scale < GROUP_MAX_EPS

    out = np.zeros((nb, 210), dtype=np.uint8)
    iscale = np.where(eps, 0.0, -128.0 / max_scale)
    d_val = np.where(eps, 0.0, 1.0 / iscale)
    out[:, 208:210] = _fp16_pack(d_val)

    sc_out = np.clip(np.rint(iscale[:, None] * scales).astype(np.int64), -128, 127).astype(np.int8)
    sc_out = np.where(eps[:, None], 0, sc_out)
    out[:, 192:208] = sc_out.astype(np.uint8)

    d = np.ascontiguousarray(out[:, 208:210]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    dj = d[:, None] * sc_out.astype(np.float32)
    L2 = np.rint(segs.reshape(nb, 16, 16) / dj[:, :, None])
    L2 = np.clip(L2, -32, 31).astype(np.int64) + 32
    L2 = L2.reshape(nb, 16 * 16)

    ql = np.zeros((nb, 128), dtype=np.uint8)
    qh = np.zeros((nb, 64), dtype=np.uint8)
    for jj in range(0, QK_K, 128):
        ql_off = (jj // 128) * 64
        qh_off = (jj // 128) * 32
        blk = L2[:, jj:jj + 128]
        for l in range(32):
            q1 = blk[:, l + 0] & 0xF
            q2 = blk[:, l + 32] & 0xF
            q3 = blk[:, l + 64] & 0xF
            q4 = blk[:, l + 96] & 0xF
            ql[:, ql_off + l + 0] = (q1 | (q3 << 4)).astype(np.uint8)
            ql[:, ql_off + l + 32] = (q2 | (q4 << 4)).astype(np.uint8)
            qh[:, qh_off + l] = ((blk[:, l + 0] >> 4) |
                                 ((blk[:, l + 32] >> 4) << 2) |
                                 ((blk[:, l + 64] >> 4) << 4) |
                                 ((blk[:, l + 96] >> 4) << 6)).astype(np.uint8)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    return out.tobytes()


def _make_qx_quants_batch(n, nmax, x):
    """make_qx_quants (rmse_type=1) for many groups at once.
    x: (G, n) float64. Returns (L (G,n) int64, scale (G,))."""
    ax = np.abs(x)
    i = np.argmax(ax, axis=1)
    rows = np.arange(x.shape[0])
    amax = ax[rows, i]
    max_ = x[rows, i]
    nz = amax >= GROUP_MAX_EPS
    iscale = np.where(nz, -nmax / np.where(max_ != 0, max_, 1.0), 0.0)
    l = _nearest_int_array(iscale[:, None] * x)
    l = np.clip(l, -nmax, nmax - 1)
    L = (l + nmax).astype(np.int64)
    w = x * x
    sumlx = np.sum(w * x * l, axis=1)
    suml2 = np.sum(w * l * l, axis=1)
    scale = np.where(suml2 != 0, sumlx / np.where(suml2 != 0, suml2, 1.0), 0.0)
    best = scale * sumlx
    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale2 = -(nmax + 0.1 * is_) / np.where(max_ != 0, max_, 1.0)
        l2 = _nearest_int_array(iscale2[:, None] * x)
        l2 = np.clip(l2, -nmax, nmax - 1)
        sumlx2 = np.sum(w * x * l2, axis=1)
        suml2_2 = np.sum(w * l2 * l2, axis=1)
        better = (suml2_2 > 0) & (sumlx2 * sumlx2 > best * suml2_2)
        if not np.any(better):
            continue
        L[better] = (l2 + nmax)[better]
        scale = np.where(better, sumlx2 / np.where(suml2_2 != 0, suml2_2, 1.0), scale)
        best = scale * sumlx2
    scale = np.where(nz, scale, 0.0)
    L = np.where(nz[:, None], L, 0)
    return L, scale


def dequantize_row_q6_k(data: bytes, n: int) -> np.ndarray:
    nb = n // QK_K
    arr = np.frombuffer(data, dtype=np.uint8, count=nb * 210).reshape(nb, 210)
    ql = arr[:, 0:128]
    qh = arr[:, 128:192]
    sc = np.ascontiguousarray(arr[:, 192:208]).view(np.int8)
    d = np.ascontiguousarray(arr[:, 208:210]).view(np.uint16).reshape(-1).view(np.float16).astype(np.float32)
    out = np.empty((nb, QK_K), dtype=np.float32)
    for nn in range(0, QK_K, 128):
        ql_off = nn // 2
        qh_off = nn // 4
        qlb = ql[:, ql_off:ql_off + 64]
        qhb = qh[:, qh_off:qh_off + 32]
        scb = sc[:, nn // 16:nn // 16 + 8]
        for l in range(32):
            is_ = l // 16
            q1 = ((qlb[:, l + 0] & 0xF) | (((qhb[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((qlb[:, l + 32] & 0xF) | (((qhb[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((qlb[:, l + 0] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((qlb[:, l + 32] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32
            out[:, nn + l] = d * scb[:, is_ + 0].astype(np.float32) * q1
            out[:, nn + l + 32] = d * scb[:, is_ + 2].astype(np.float32) * q2
            out[:, nn + l + 64] = d * scb[:, is_ + 4].astype(np.float32) * q3
            out[:, nn + l + 96] = d * scb[:, is_ + 6].astype(np.float32) * q4
    return out.reshape(n)
