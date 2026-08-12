"""Tensor data type conversions and cluster quantizers.

Port of src/DataTransform.zig (Quantizer) plus the tensor-level helpers from
TensorClusters.zig needed to write ComfyUI cluster formats. All quantize paths
return raw bytes matching the reference implementations.
"""

from __future__ import annotations

import math
import struct
from typing import Optional, Tuple

import numpy as np

from . import ggml as ggml_mod
from . import types as types_mod


# ---------------------------------------------------------------------------
# fp16 / bf16
# ---------------------------------------------------------------------------

def f32_to_bf16(x) -> int:
    bits = np.uint32(np.float32(x).view(np.uint32))
    return int(np.uint16(bits >> np.uint32(16)))


def bf16_to_f32(x: int) -> float:
    bits = np.uint32(x) << np.uint32(16)
    return float(np.uint32(bits).view(np.float32))


# ---------------------------------------------------------------------------
# FP8 E4M3 / E5M2
# ---------------------------------------------------------------------------

def fp8_e4m3_to_f32(x: int) -> float:
    sign = (x >> 7) & 0x1
    exp = (x >> 3) & 0xF
    mant = x & 0x7
    sign_mult = 1.0 - 2.0 * sign
    if exp == 0:
        return sign_mult * (mant / 8.0) * math.exp2(-6.0)
    if exp == 0xF and mant == 0x7:
        return math.nan
    e = exp - 7.0
    m = 1.0 + mant / 8.0
    return sign_mult * m * math.exp2(e)


def fp8_e5m2_to_f32(x: int) -> float:
    sign = (x >> 7) & 0x1
    exp = (x >> 2) & 0x1F
    mant = x & 0x3
    if exp == 0:
        m = mant / 4.0
        return (1.0 - 2.0 * sign) * m * math.exp2(-14.0)
    if exp == 0x1F:
        if mant == 0:
            return math.inf * (1.0 - 2.0 * sign)
        return math.nan
    e = exp - 15.0
    m = 1.0 + mant / 4.0
    return (1.0 - 2.0 * sign) * m * math.exp2(e)


LUT_E4M3 = np.array([fp8_e4m3_to_f32(i) for i in range(256)], dtype=np.float32)
LUT_E5M2 = np.array([fp8_e5m2_to_f32(i) for i in range(256)], dtype=np.float32)


def _clz(n: int) -> int:
    """Leading zero count of a 32-bit value (zig @clz on u32)."""
    if n == 0:
        return 32
    return 32 - n.bit_length()


def _f32_bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def f32_to_fp8_e4m3(x) -> np.ndarray:
    """Vectorized scalar ml_dtypes float8_e4m3fn ConvertFrom<float>."""
    scalar = np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        x = x.reshape(1)
    out = np.empty(x.shape, dtype=np.uint8)
    flat = x.ravel()
    flat_bits = flat.view(np.uint32)
    sign = (flat_bits >> np.uint32(31)).astype(np.uint32)
    abs_bits = (flat_bits & np.uint32(0x7FFFFFFF)).astype(np.uint32)

    special = abs_bits >= np.uint32(0x7F800000)
    zero = abs_bits == np.uint32(0)

    biased = (abs_bits >> np.uint32(23)).astype(np.uint32)
    frac = (abs_bits & np.uint32(0x7FFFFF)).astype(np.uint32)

    # normal path
    unbiased = biased.astype(np.int64) - 127
    norm_mant = (np.uint32(0x800000) | frac).astype(np.uint32)

    tbe = unbiased + 6
    denorm_adj = np.maximum(0, -tbe)
    ashift = np.minimum(20 + denorm_adj, 25).astype(np.int64)
    roundoff = ashift
    bias = ((norm_mant.astype(np.int64) >> roundoff) & 1) + (np.int64(1) << (roundoff - 1)) - 1
    rounded = norm_mant.astype(np.int64) + bias
    aligned = (rounded >> roundoff).astype(np.uint32) & np.uint32(0xFF)
    exp_bits = np.maximum(0, tbe).astype(np.uint32)
    result_normal = ((aligned + (exp_bits << np.uint32(3))) & np.uint32(0xFF)).astype(np.uint32)

    # subnormal path (rare): RTE(|x| * 512) via add-magic
    cap = np.minimum(abs_bits, np.uint32(0x3C800000))
    capped = cap.view(np.float32)
    magic = np.float32(2.0 ** 23)
    subnorm_mant = ((capped * np.float32(512.0) + magic - magic).view(np.uint32)).astype(np.uint32)

    subnorm = (tbe < 0) & ~zero & ~special
    result = np.where(subnorm, subnorm_mant, result_normal)

    overflow = (tbe >= 16) | (result > np.uint32(0x7E))
    result = np.where(overflow, np.uint32(0x7F), result)

    out_flat = ((sign << np.uint32(7)) | result).astype(np.uint8)
    out_flat = np.where(special, (sign << np.uint32(7)) | np.uint32(0x7F), out_flat)
    out.ravel()[:] = out_flat
    if scalar:
        return out.reshape(())
    return out


def f32_to_fp8_e5m2(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty(x.shape, dtype=np.uint8)
    flat = x.ravel()
    flat_bits = flat.view(np.uint32)
    sign = (flat_bits >> np.uint32(31)).astype(np.uint32)
    abs_bits = (flat_bits & np.uint32(0x7FFFFFFF)).astype(np.uint32)

    is_nan = abs_bits > np.uint32(0x7F800000)
    is_inf = abs_bits == np.uint32(0x7F800000)
    zero = abs_bits == np.uint32(0)

    biased = (abs_bits >> np.uint32(23)).astype(np.uint32)
    frac = (abs_bits & np.uint32(0x7FFFFF)).astype(np.uint32)
    unbiased = biased.astype(np.int64) - 127
    norm_mant = (np.uint32(0x800000) | frac).astype(np.uint32)

    tbe = unbiased + 14
    denorm_adj = np.maximum(0, -tbe)
    ashift = np.minimum(21 + denorm_adj, 25).astype(np.int64)
    roundoff = ashift
    bias = ((norm_mant.astype(np.int64) >> roundoff) & 1) + (np.int64(1) << (roundoff - 1)) - 1
    rounded = norm_mant.astype(np.int64) + bias
    aligned = (rounded >> roundoff).astype(np.uint32) & np.uint32(0xFF)
    exp_bits = np.maximum(0, tbe).astype(np.uint32)
    result_normal = ((aligned + (exp_bits << np.uint32(2))) & np.uint32(0xFF)).astype(np.uint32)

    cap = np.minimum(abs_bits, np.uint32(0x38800000))
    capped = cap.view(np.float32)
    magic = np.float32(2.0 ** 23)
    subnorm_mant = ((capped * np.float32(65536.0) + magic - magic).view(np.uint32)).astype(np.uint32)

    subnorm = (tbe < 0) & ~zero & ~is_nan & ~is_inf
    result = np.where(subnorm, subnorm_mant, result_normal)

    overflow = (tbe >= 31) | (result > np.uint32(0x7B))
    result = np.where(overflow, np.uint32(0x7C), result)

    out_flat = ((sign << np.uint32(7)) | result).astype(np.uint8)
    out_flat = np.where(is_inf, (sign << np.uint32(7)) | np.uint32(0x7C), out_flat)
    out_flat = np.where(is_nan, (sign << np.uint32(7)) | np.uint32(0x7E), out_flat)
    out.ravel()[:] = out_flat
    return out


# ---------------------------------------------------------------------------
# E8M0
# ---------------------------------------------------------------------------

def e8m0_to_f32(x: int) -> np.float32:
    if x == 0:
        return np.float32(2.0 ** -127)
    if x == 255:
        return np.float32(np.nan)
    return np.uint32(x) << np.uint32(23)


def f32_to_e8m0(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    abs_bits = (bits & np.uint32(0x7FFFFFFF)).astype(np.uint32)
    biased = (abs_bits >> np.uint32(23)).astype(np.uint32)
    out = np.zeros(x.shape, dtype=np.uint8)
    normal = (biased > np.uint32(0)) & (biased < np.uint32(255)) & (abs_bits != np.uint32(0))
    out[normal] = biased[normal].astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# FP4 E2M1
# ---------------------------------------------------------------------------

FP4_POSITIVES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
LUT_FP4_E2M1 = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def fp4_e2m1_to_f32(nibble: int) -> float:
    positives = FP4_POSITIVES
    sign = -1.0 if (nibble >> 3) != 0 else 1.0
    return sign * float(positives[nibble & 0x7])


def f32_to_fp4_e2m1(x) -> np.ndarray:
    """Round-to-nearest-even over the 8 representable magnitudes."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    sign = (bits >> np.uint32(31)).astype(np.uint32) & np.uint32(1)
    abs_bits = (bits & np.uint32(0x7FFFFFFF)).astype(np.uint32)
    nan_inf = abs_bits >= np.uint32(0x7F800000)
    absv = abs_bits.view(np.float32)
    code = np.where(absv <= np.float32(0.25), 0,
        np.where(absv < np.float32(0.75), 1,
        np.where(absv <= np.float32(1.25), 2,
        np.where(absv < np.float32(1.75), 3,
        np.where(absv <= np.float32(2.5), 4,
        np.where(absv < np.float32(3.5), 5,
        np.where(absv <= np.float32(5.0), 6, 7))))))).astype(np.uint32)
    code = np.where(nan_inf, np.uint32(7), code)
    return ((sign << np.uint32(3)) | code).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stochastic rounding
# ---------------------------------------------------------------------------

def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def stochastic_uniform(seed: int, idx: int) -> float:
    h = splitmix64(seed ^ splitmix64(idx))
    return (h >> 40) * (2.0 ** -24)


def round_half_to_even(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    fl = np.floor(x)
    diff = x - fl
    out = np.where(diff < 0.5, fl,
                   np.where(diff > 0.5, fl + 1.0,
                            np.where(np.mod(fl, 2.0) == 0.0, fl, fl + 1.0)))
    return out


# ---------------------------------------------------------------------------
# Hadamard rotation
# ---------------------------------------------------------------------------

H4_RAW = np.array([1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1],
                  dtype=np.float32)


def is_valid_hadamard_size(size: int) -> bool:
    if size < 4:
        return False
    if (size & (size - 1)) != 0:
        return False
    ctz = (size & -size).bit_length() - 1
    return (ctz & 1) == 0


def hadamard_transform_in_place(v: np.ndarray) -> None:
    n = v.size
    h = 1
    while h < n:
        for i in range(0, n, h * 4):
            for j in range(i, i + h):
                a = v[j]
                b = v[j + h]
                c = v[j + 2 * h]
                d = v[j + 3 * h]
                v[j] = a + b + c - d
                v[j + h] = a + b - c + d
                v[j + 2 * h] = a - b + c + d
                v[j + 3 * h] = -a + b + c + d
        h *= 4
    norm = 1.0 / math.sqrt(n)
    v *= norm


def rotate_groupwise_in_place(buf: np.ndarray, rows: int, cols: int,
                              group_size: int) -> None:
    if not is_valid_hadamard_size(group_size):
        raise ValueError("InvalidHadamardSize")
    if cols % group_size != 0:
        raise ValueError("ColsNotDivisibleByGroupSize")
    n_groups = cols // group_size
    buf = buf.reshape(rows, cols)
    for row in range(rows):
        for g in range(n_groups):
            base = row * cols + g * group_size
            hadamard_transform_in_place(buf.ravel()[base:base + group_size])


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert_tensor_data(src_data: bytes, src_type: str, dst_type: str,
                        element_count: int, pool: Optional[object] = None) -> bytes:
    if types_mod.equivalent_type(src_type, dst_type):
        return src_data

    f32_buffer = dequantize_to_f32(src_data, src_type, element_count, pool)
    return quantize_from_f32(f32_buffer, dst_type, element_count, pool)


def dequantize_to_f32(input_bytes: bytes, src_type: str, element_count: int,
                      pool=None) -> np.ndarray:
    return _dequantize_to_f32(input_bytes, src_type, element_count, pool)


def _dequantize_to_f32(input_bytes: bytes, src_type: str, element_count: int,
                       pool=None) -> np.ndarray:
    if src_type in ("F8_E4M3",):
        if len(input_bytes) != element_count:
            raise ValueError("InputSizeMismatch")
        arr = np.frombuffer(input_bytes, dtype=np.uint8, count=element_count)
        return LUT_E4M3[arr].astype(np.float32)
    if src_type in ("F8_E5M2",):
        if len(input_bytes) != element_count:
            raise ValueError("InputSizeMismatch")
        arr = np.frombuffer(input_bytes, dtype=np.uint8, count=element_count)
        return LUT_E5M2[arr].astype(np.float32)
    if src_type in ("F4_E2M1", "MXFP4"):
        if len(input_bytes) * 2 != element_count:
            raise ValueError("InputSizeMismatch")
        return dequantize_fp4(input_bytes, element_count)
    if src_type == "mxfp4":
        n_blocks = len(input_bytes) // 17
        if n_blocks * 32 != element_count:
            raise ValueError("InputSizeMismatch")
        return ggml_mod.dequantize_row_mxfp4(input_bytes, element_count)
    if src_type in ("BF16", "bf16"):
        if len(input_bytes) // 2 != element_count:
            raise ValueError("InputSizeMismatch")
        u16 = np.frombuffer(input_bytes, dtype=np.uint16, count=element_count)
        bits = u16.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32).astype(np.float32)
    if src_type in ("F16", "f16"):
        if len(input_bytes) // 2 != element_count:
            raise ValueError("InputSizeMismatch")
        return np.frombuffer(input_bytes, dtype=np.float16,
                             count=element_count).astype(np.float32)
    if src_type in ("F32", "f32"):
        return np.frombuffer(input_bytes, dtype=np.float32,
                             count=element_count).copy()
    if src_type in ("F64", "f64"):
        if len(input_bytes) // 8 != element_count:
            raise ValueError("InputSizeMismatch")
        return np.frombuffer(input_bytes, dtype=np.float64,
                             count=element_count).astype(np.float32)
    # Generic GGUF block types
    if types_mod.format_type(src_type) != types_mod.FileType.GGUF:
        raise ValueError(f"UnsupportedSourceType: {src_type}")
    return ggml_mod.dequantize_row(src_type, input_bytes, element_count)


def quantize_from_f32(input_f32: np.ndarray, dst_type: str, element_count: int,
                      pool=None) -> bytes:
    input_f32 = np.ascontiguousarray(input_f32, dtype=np.float32)
    if dst_type in ("f32", "F32"):
        return input_f32.tobytes()
    if dst_type in ("BF16", "bf16"):
        bits = input_f32.view(np.uint32)
        return (bits >> np.uint32(16)).astype(np.uint16).tobytes()
    if dst_type in ("f16", "F16"):
        return input_f32.astype(np.float16).view(np.uint16).tobytes()
    if dst_type in ("F8_E4M3", "F8_E5M2"):
        if len(input_f32) != element_count:
            raise ValueError("OutputBufferSizeMismatch")
        if dst_type == "F8_E4M3":
            return f32_to_fp8_e4m3(input_f32).tobytes()
        return f32_to_fp8_e5m2(input_f32).tobytes()
    if dst_type in ("F4_E2M1", "MXFP4"):
        if len(input_f32) * 2 != element_count:
            raise ValueError("OutputBufferSizeMismatch")
        return quantize_fp4(input_f32)
    if dst_type in ("q8_0", "q5_0", "q4_0", "q5_1", "q4_1", "q6_k", "q5_k",
                    "q4_k", "q3_k", "q2_k", "mxfp4"):
        return ggml_mod.quantize_row(dst_type, input_f32)
    raise ValueError(f"UnsupportedDestinationType: {dst_type}")


# ---------------------------------------------------------------------------
# FP4 packing (safetensors: element[2i] low nibble, element[2i+1] high nibble)
# ---------------------------------------------------------------------------

def dequantize_fp4(input_bytes: bytes, element_count: int) -> np.ndarray:
    arr = np.frombuffer(input_bytes, dtype=np.uint8)
    n_pairs = element_count // 2
    lo = (arr[:n_pairs] & 0x0F).astype(np.uint32)
    hi = (arr[:n_pairs] >> 4).astype(np.uint32)
    out = np.empty(element_count, dtype=np.float32)
    out[0::2] = LUT_FP4_E2M1[lo]
    out[1::2] = LUT_FP4_E2M1[hi]
    if element_count % 2 == 1:
        out[-1] = LUT_FP4_E2M1[arr[n_pairs] & 0x0F]
    return out


def quantize_fp4(input_f32: np.ndarray) -> bytes:
    n = input_f32.size
    codes = f32_to_fp4_e2m1(input_f32).astype(np.uint32)
    if n % 2 == 1:
        padded = np.append(codes, np.uint32(0))
        n_use = n
    else:
        padded = codes
        n_use = n
    n_pairs = n_use // 2
    lo = padded[0:n_use:2]
    hi = padded[1:n_use:2]
    return ((hi << 4) | lo).astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# ComfyUI FP8 cluster
# ---------------------------------------------------------------------------

def quantize_to_comfy_fp8(input: np.ndarray) -> Tuple[np.ndarray, np.float32]:
    input = np.asarray(input, dtype=np.float32)
    amax = float(np.max(np.abs(input))) if input.size else 0.0
    fp8_max = 448.0
    scale = np.float32(amax / fp8_max) if amax > 0.0 else np.float32(1.0)
    inv_scale = np.float32(1.0) / scale
    scaled = (input * inv_scale).astype(np.float32)
    weight = f32_to_fp8_e4m3(scaled)
    return weight, scale


# ---------------------------------------------------------------------------
# INT8 / ConvRot
# ---------------------------------------------------------------------------

def quantize_to_int8(input: np.ndarray, rows: int, cols: int, convrot: bool,
                     group_size: int, stochastic_rounding: int = 0,
                     pool=None) -> Tuple[np.ndarray, np.ndarray]:
    input = np.ascontiguousarray(input, dtype=np.float32)
    if input.size != rows * cols:
        raise ValueError("InputSizeMismatch")
    work = input.copy()
    if convrot:
        rotate_groupwise_in_place(work, rows, cols, group_size)
    weight = np.empty(rows * cols, dtype=np.uint8)
    scale = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        row = work[r * cols:(r + 1) * cols]
        finite = row[np.isfinite(row)]
        amax = float(np.max(np.abs(finite))) if finite.size else 0.0
        s = max(amax / 127.0, 1e-30)
        scale[r] = s
        q = np.clip(round_half_to_even(row / s), -128.0, 127.0)
        weight[r * cols:(r + 1) * cols] = q.astype(np.int8).view(np.uint8)
    return weight, scale


def quantize_to_convrot_int8(input, rows, cols, group_size, pool=None):
    return quantize_to_int8(input, rows, cols, True, group_size, pool=pool)


# ---------------------------------------------------------------------------
# INT4 ConvRot
# ---------------------------------------------------------------------------

def _quantize_int4_nibble(v, s, seed, idx) -> int:
    scaled = v / s
    if seed == 0:
        rounded = round_half_to_even(scaled)
    else:
        rounded = math.floor(scaled + stochastic_uniform(seed, idx))
    q = max(-7.0, min(7.0, rounded))
    return int(np.uint8(np.int8(q))) & 0x0F


def quantize_to_int4(input: np.ndarray, rows: int, cols: int, convrot: bool,
                     group_size: int, stochastic_rounding: int = 0,
                     pool=None) -> Tuple[np.ndarray, np.ndarray]:
    input = np.ascontiguousarray(input, dtype=np.float32)
    if input.size != rows * cols:
        raise ValueError("InputSizeMismatch")
    if cols % 2 != 0:
        raise ValueError("ColsNotEven")
    work = input.copy()
    if convrot:
        rotate_groupwise_in_place(work, rows, cols, group_size)
    packed_cols = cols // 2
    weight = np.empty(rows * packed_cols, dtype=np.uint8)
    scale = np.empty(rows, dtype=np.float32)
    seed = stochastic_rounding
    for r in range(rows):
        row = work[r * cols:(r + 1) * cols]
        finite = row[np.isfinite(row)]
        amax = float(np.max(np.abs(finite))) if finite.size else 0.0
        s = max(amax / 7.0, 1e-30)
        scale[r] = s
        row_base = r * cols
        for pc in range(packed_cols):
            lo = _quantize_int4_nibble(float(row[2 * pc]), s, seed, row_base + 2 * pc)
            hi = _quantize_int4_nibble(float(row[2 * pc + 1]), s, seed, row_base + 2 * pc + 1)
            weight[r * packed_cols + pc] = lo | (hi << 4)
    return weight, scale


def quantize_to_convrot_int4(input, rows, cols, group_size,
                             stochastic_rounding: int, pool=None):
    return quantize_to_int4(input, rows, cols, True, group_size,
                            stochastic_rounding, pool=pool)


# ---------------------------------------------------------------------------
# ASYM W4A8
# ---------------------------------------------------------------------------

W4A8_GROUP_SIZE = 16
W4A8_CODEBOOK = np.array([
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187,
    -0.155210, -0.050720, 0.052541, 0.156985, 0.265284, 0.379533,
    0.502636, 0.638953, 0.794876, 0.980671,
], dtype=np.float32)
W4A8_GATE_KURTOSIS = -0.1
W4A8_GATE_SAMPLE_ELEMS = 1 << 19


def _level_midpoints(levels: np.ndarray) -> np.ndarray:
    return (levels[:-1] + levels[1:]) * 0.5


W4A8_CODEBOOK_MIDS = _level_midpoints(W4A8_CODEBOOK)


def _assign_group(vals: np.ndarray, mids: np.ndarray) -> np.ndarray:
    count = np.zeros(vals.size, dtype=np.uint8)
    for j in range(mids.size):
        count += (mids[j] < vals).astype(np.uint8)
    return count


def _group_amax(gv: np.ndarray) -> float:
    av = np.abs(gv)
    av = np.where(av < np.float32(np.finfo(np.float32).max), av, np.float32(0.0))
    return float(np.max(av)) if av.size else 0.0


def _w4a8_codebook_fit_recommended(rotated: np.ndarray, cols: int) -> bool:
    n_groups = rotated.size // W4A8_GROUP_SIZE
    if n_groups == 0 or cols == 0:
        return False
    want_groups = max(1, W4A8_GATE_SAMPLE_ELEMS // W4A8_GROUP_SIZE)
    stride = max(1, n_groups // want_groups)
    groups = rotated.reshape(-1, W4A8_GROUP_SIZE)
    sampled = groups[::stride]
    amax = np.max(np.abs(sampled), axis=1)
    gs = np.maximum(amax, 1e-8)[:, None]
    normed = (sampled / gs).astype(np.float64)
    mean = np.mean(normed)
    d = normed - mean
    m2 = np.sum(d * d)
    m4 = np.sum(d * d * d * d)
    count = float(normed.size)
    if count < 2.0:
        return False
    variance = m2 / (count - 1.0)
    sd = math.sqrt(variance) + 1e-9
    excess = (m4 / count) / (sd ** 4) - 3.0
    return excess > W4A8_GATE_KURTOSIS


def quantize_to_asym_w4a8(input: np.ndarray, rows: int, cols: int,
                          convrot_group_size: int,
                          pool=None):
    input = np.ascontiguousarray(input, dtype=np.float32)
    if input.size != rows * cols:
        raise ValueError("InputSizeMismatch")
    if cols % W4A8_GROUP_SIZE != 0:
        raise ValueError("ColsNotDivisibleByGroupSize")
    if cols % convrot_group_size != 0:
        raise ValueError("ColsNotDivisibleByGroupSize")
    rotated = input.copy()
    rotate_groupwise_in_place(rotated, rows, cols, convrot_group_size)
    groups_per_row = cols // W4A8_GROUP_SIZE
    weight = np.empty(rows * (cols // 2), dtype=np.uint8)
    s_rel = np.empty(rows * groups_per_row, dtype=np.uint8)
    s_channel = np.empty(rows, dtype=np.float32)
    group_scales = np.empty(rows * groups_per_row, dtype=np.float32)
    cb = W4A8_CODEBOOK
    for r in range(rows):
        row = rotated[r * cols:(r + 1) * cols]
        row_scales = group_scales[r * groups_per_row:(r + 1) * groups_per_row]
        max_shifted = 0.0
        for g in range(groups_per_row):
            grp = row[g * W4A8_GROUP_SIZE:(g + 1) * W4A8_GROUP_SIZE]
            gv = grp
            gs = max(_group_amax(gv), 1e-8)
            idx = _assign_group(gv / gs, W4A8_CODEBOOK_MIDS)
            for _ in range(2):
                c = cb[idx]
                num = float(np.sum(gv * c))
                den = float(np.sum(c * c))
                gs = max(num / max(den, 1e-8), 1e-8)
                idx = _assign_group(gv / gs, W4A8_CODEBOOK_MIDS)
            row_scales[g] = gs
            max_shifted = max(max_shifted, float(np.max(np.abs(cb[idx] * gs))))
        sc = max(max_shifted / 127.0, 1e-8)
        s_channel[r] = sc
        for g in range(groups_per_row):
            gv = row[g * W4A8_GROUP_SIZE:(g + 1) * W4A8_GROUP_SIZE]
            s_rel_byte = int(np.asarray(f32_to_fp8_e4m3(row_scales[g] / sc)).reshape(-1)[0])
            s_rel[r * groups_per_row + g] = s_rel_byte
            s_rel_dec = fp8_e4m3_to_f32(s_rel_byte)
            levels = np.clip(round_half_to_even(cb * s_rel_dec), -127.0, 127.0)
            mids = _level_midpoints(levels)
            idx = _assign_group(gv / sc, mids)
            base = g * W4A8_GROUP_SIZE
            for p in range(W4A8_GROUP_SIZE // 2):
                flat = base + 2 * p
                weight[r * (cols // 2) + flat // 2] = idx[2 * p] | (idx[2 * p + 1] << 4)
    recommended = _w4a8_codebook_fit_recommended(rotated, cols)
    return weight, s_rel, s_channel, recommended


# ---------------------------------------------------------------------------
# ComfyUI MXFP4 / MXFP8 clusters
# ---------------------------------------------------------------------------

def quantize_to_comfy_mxfp4(input: np.ndarray, pool=None):
    input = np.ascontiguousarray(input, dtype=np.float32)
    n = input.size
    if n % 32 != 0:
        raise ValueError("ElementCountNotMultipleOf32")
    n_blocks = n // 32
    gguf_buf = ggml_mod.quantize_row_mxfp4(input)
    weight = np.empty(n // 2, dtype=np.uint8)
    scale = np.empty(n_blocks, dtype=np.uint8)
    for bi in range(n_blocks):
        block = gguf_buf[bi * 17:bi * 17 + 17]
        scale[bi] = block[0]
        nibbles = np.empty(32, dtype=np.uint8)
        for j in range(16):
            nibbles[j] = block[1 + j] & 0xF
            nibbles[j + 16] = block[1 + j] >> 4
        base = bi * 16
        for k in range(16):
            weight[base + k] = nibbles[2 * k] | (nibbles[2 * k + 1] << 4)
    return weight, scale


def quantize_to_comfy_mxfp8(input: np.ndarray, pool=None):
    input = np.ascontiguousarray(input, dtype=np.float32)
    n_elements = input.size
    if n_elements % 32 != 0:
        raise ValueError("InvalidMxfp8Size")
    n_blocks = n_elements // 32
    weight = np.empty(n_elements, dtype=np.uint8)
    scale = np.empty(n_blocks, dtype=np.uint8)
    blocks = input.reshape(n_blocks, 32)
    for bi in range(n_blocks):
        block = blocks[bi]
        amax = float(np.max(np.abs(block))) if block.size else 0.0
        if amax < 1e-30:
            scale[bi] = 0
            weight[bi * 32:(bi + 1) * 32] = 0
            continue
        log2_amax = math.log2(amax)
        shared_exp_unbiased = int(math.floor(log2_amax)) + 1
        shared_exp_biased = shared_exp_unbiased + 127
        scale_byte = max(0, min(254, shared_exp_biased))
        scale[bi] = scale_byte
        scale_f32 = float(e8m0_to_f32(scale_byte))
        inv_scale = 1.0 / scale_f32
        scaled = block * inv_scale
        weight[bi * 32:(bi + 1) * 32] = f32_to_fp8_e4m3(scaled)
    return weight, scale


def to_blocked_mxfp8(scale_raw: np.ndarray, n_rows: int, n_scale_cols: int) -> np.ndarray:
    n_row_blocks = (n_rows + 127) // 128
    n_col_blocks = (n_scale_cols + 3) // 4
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    out = np.zeros(padded_rows * padded_cols, dtype=np.uint8)
    for r in range(n_rows):
        rb = r // 128
        r_within = r % 128
        a = r_within // 32
        b = r_within % 32
        for c in range(n_scale_cols):
            cb = c // 4
            c_within = c % 4
            ik = rb * n_col_blocks + cb
            flat = ik * 512 + b * 16 + a * 4 + c_within
            out[flat] = scale_raw[r * n_scale_cols + c]
    return out
