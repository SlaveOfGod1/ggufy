"""ComfyUI quantized-tensor cluster grouping, dequantization and writing.

Port of src/TensorClusters.zig. A "cluster" is a layer stored as several
physical sub-tensors: a weight plus scale tensor(s) plus a `.comfy_quant`
identity marker (or a file-level `_quantization_metadata` header). The
conversion pipeline collapses clusters into a single logical tensor before
assigning output types, then expands them again when writing safetensors.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import data_transform as dt
from . import types as types_mod

COMFY_SUFFIX = ".comfy_quant"

FP8_COMFY_JSON = b'{"format": "float8_e4m3fn"}'
MXFP4_COMFY_JSON = b'{"format":"mxfp4"}'
MXFP8_COMFY_JSON = b'{"format":"mxfp8"}'
NVFP4_COMFY_JSON = b'{"format": "nvfp4"}'
INT8_COMFY_JSON = b'{"per_row": true, "format": "int8_tensorwise"}'
INT8_CONVROT_COMFY_JSON = b'{"convrot": true, "convrot_groupsize": 256, "per_row": true, "format": "int8_tensorwise"}'
INT4_CONVROT_COMFY_JSON = b'{"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64, "linear_dtype": "int4"}'
ASYM_W4A8_COMFY_JSON = b'{"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}'

INT8_CONVROT_GROUP_SIZE = 256
INT4_CONVROT_GROUP_SIZE = 256
ASYM_W4A8_CONVROT_GROUP_SIZE = 256
ASYM_W4A8_GROUP_SIZE = dt.W4A8_GROUP_SIZE
DEFAULT_STOCHASTIC_SEED = 0xC0FFEE

COMFY_QUANT_SCHEMES = {
    "nvfp4": "nvfp4",
    "mxfp8_e4m3fn": "mxfp8_e4m3fn",
    "mxfp8": "mxfp8_e4m3fn",
    "mxfp4": "mxfp4",
    "float8_e4m3fn": "float8_e4m3fn",
    "int8_tensorwise": "int8_convrot",
    "convrot_w4a4": "convrot_w4a4",
    "asym_w4a8_int8": "asym_w4a8_int8",
}


COMQUANT_SCHEME_MAP = {
    "nvfp4": "nvfp4",
    "mxfp8_e4m3fn": "mxfp8_e4m3fn",
    "mxfp8": "mxfp8_e4m3fn",
    "mxfp4": "mxfp4",
    "float8_e4m3fn": "float8_e4m3fn",
    "int8_tensorwise": "int8_convrot",
    "convrot_w4a4": "convrot_w4a4",
    "asym_w4a8_int8": "asym_w4a8_int8",
}


def parse_comfy_quant_scheme(data: bytes) -> str:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "unknown"
    if not isinstance(parsed, dict):
        return "unknown"
    fmt = parsed.get("format")
    if not isinstance(fmt, str):
        return "unknown"
    return COMQUANT_SCHEME_MAP.get(fmt, "unknown")


def _comfy_quant_bool(data: bytes, key: str, default: bool) -> bool:
    try:
        parsed = json.loads(data.decode("utf-8"))
        if isinstance(parsed, dict):
            v = parsed.get(key)
            if isinstance(v, bool):
                return v
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return default


def _comfy_quant_int(data: bytes, key: str, default: int) -> int:
    try:
        parsed = json.loads(data.decode("utf-8"))
        if isinstance(parsed, dict):
            v = parsed.get(key)
            if isinstance(v, int) and v > 0:
                return v
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return default


def name_suffix_match(full_name: str, stripped: str) -> bool:
    if full_name == stripped:
        return True
    return (len(full_name) > len(stripped)
            and full_name[len(full_name) - len(stripped) - 1] == '.'
            and full_name.endswith(stripped))


def best_source_match(tensors: List[Any], name: str) -> Optional[int]:
    best = None
    for i, t in enumerate(tensors):
        if t.name == name:
            return i
        if not name_suffix_match(t.name, name):
            continue
        if best is None or len(t.name) < len(tensors[best].name):
            best = i
    return best


def _read_tensor_bytes(source, tensor, size: Optional[int] = None) -> bytes:
    n = size if size is not None else tensor.size
    f = source.open_file_for_tensor(tensor.name)
    f.seek(tensor.offset + source.current_data_begin)
    return f.read(n)


# ---------------------------------------------------------------------------
# Cluster structures
# ---------------------------------------------------------------------------

@dataclass
class Fp4Cluster:
    base_name: str
    weight: Any
    weight_scale: Any
    weight_scale_2: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int


@dataclass
class Float8Cluster:
    base_name: str
    weight: Any
    weight_scale: Any
    input_scale: Optional[Any]
    comfy_quant: Optional[Any]
    rows: int
    cols: int


@dataclass
class Mxfp4Cluster:
    base_name: str
    weight: Any
    weight_scale: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int


@dataclass
class Mxfp8Cluster:
    base_name: str
    weight: Any
    weight_scale: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int


@dataclass
class Int8ConvrotCluster:
    base_name: str
    weight: Any
    weight_scale: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int
    convrot: bool
    group_size: int


@dataclass
class Int4Cluster:
    base_name: str
    weight: Any
    weight_scale: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int
    convrot: bool
    group_size: int


@dataclass
class AsymW4a8Cluster:
    base_name: str
    weight: Any
    weight_s_rel: Any
    weight_s_channel: Any
    weight_codebook: Any
    comfy_quant: Optional[Any]
    rows: int
    cols: int
    group_size: int
    convrot_group_size: int


@dataclass
class GroupResult:
    fp4_clusters: List[Fp4Cluster] = field(default_factory=list)
    float8_clusters: List[Float8Cluster] = field(default_factory=list)
    mxfp4_clusters: List[Mxfp4Cluster] = field(default_factory=list)
    mxfp8_clusters: List[Mxfp8Cluster] = field(default_factory=list)
    int8_convrot_clusters: List[Int8ConvrotCluster] = field(default_factory=list)
    int4_clusters: List[Int4Cluster] = field(default_factory=list)
    asym_w4a8_clusters: List[AsymW4a8Cluster] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cluster grouping
# ---------------------------------------------------------------------------

def _append_int8_convrot_cluster(source, name_map, base_name, convrot,
                                 group_size, comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for int8_convrot cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for int8_convrot cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    if len(weight.dims) != 2:
        print(f"TensorClusters: int8_convrot cluster {base_name} weight is not 2-D; skipping")
        return False
    rows = weight.dims[0]
    cols = weight.dims[1]
    if weight_scale.size != rows * 4:
        print(f"TensorClusters: int8_convrot cluster {base_name} has non-per-row scale "
              f"({weight_scale.size} bytes, expected {rows * 4}); skipping")
        return False
    if convrot and (not dt.is_valid_hadamard_size(group_size) or cols % group_size != 0):
        print(f"TensorClusters: int8_convrot cluster {base_name} has incompatible "
              f"group_size {group_size} for cols {cols}; skipping")
        return False
    groups.int8_convrot_clusters.append(Int8ConvrotCluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        comfy_quant=comfy_quant, rows=rows, cols=cols, convrot=convrot,
        group_size=group_size))
    return True


def _append_int4_cluster(source, name_map, base_name, convrot, group_size,
                         comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for int4 cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for int4 cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    if len(weight.dims) != 2:
        print(f"TensorClusters: int4 cluster {base_name} weight is not 2-D; skipping")
        return False
    rows = weight.dims[0]
    cols = weight.dims[1] * 2
    if weight_scale.size != rows * 4:
        print(f"TensorClusters: int4 cluster {base_name} has non-per-row scale "
              f"({weight_scale.size} bytes, expected {rows * 4}); skipping")
        return False
    if convrot and (not dt.is_valid_hadamard_size(group_size) or cols % group_size != 0):
        print(f"TensorClusters: int4 cluster {base_name} has incompatible "
              f"group_size {group_size} for cols {cols}; skipping")
        return False
    groups.int4_clusters.append(Int4Cluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        comfy_quant=comfy_quant, rows=rows, cols=cols, convrot=convrot,
        group_size=group_size))
    return True


def _append_asym_w4a8_cluster(source, name_map, base_name, group_size,
                              convrot_group_size, comfy_quant, groups):
    suffixes = [".weight", ".weight_s_rel", ".weight_s_channel", ".weight_codebook"]
    idxs = []
    for s in suffixes:
        name = base_name + s
        if name not in name_map:
            print(f"TensorClusters: missing {s} for asym_w4a8 cluster {base_name}")
            return False
        idxs.append(name_map[name])
    weight = source.tensors[idxs[0]]
    s_rel = source.tensors[idxs[1]]
    s_channel = source.tensors[idxs[2]]
    codebook = source.tensors[idxs[3]]
    if len(weight.dims) != 2:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} weight is not 2-D; skipping")
        return False
    rows = weight.dims[0]
    cols = weight.dims[1] * 2
    if group_size == 0 or cols % group_size != 0:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} has incompatible "
              f"group_size {group_size} for cols {cols}; skipping")
        return False
    n_groups = cols // group_size
    if s_rel.size != rows * n_groups:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} has wrong s_rel size "
              f"({s_rel.size} bytes, expected {rows * n_groups}); skipping")
        return False
    if s_channel.size != rows * 4:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} has wrong s_channel size "
              f"({s_channel.size} bytes, expected {rows * 4}); skipping")
        return False
    if codebook.size != 16 * 4:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} has wrong codebook size "
              f"({codebook.size} bytes, expected 64); skipping")
        return False
    if not dt.is_valid_hadamard_size(convrot_group_size) or cols % convrot_group_size != 0:
        print(f"TensorClusters: asym_w4a8 cluster {base_name} has incompatible "
              f"convrot_groupsize {convrot_group_size} for cols {cols}; skipping")
        return False
    groups.asym_w4a8_clusters.append(AsymW4a8Cluster(
        base_name=base_name, weight=weight, weight_s_rel=s_rel,
        weight_s_channel=s_channel, weight_codebook=codebook,
        comfy_quant=comfy_quant, rows=rows, cols=cols, group_size=group_size,
        convrot_group_size=convrot_group_size))
    return True


def _append_nvfp4_cluster(source, name_map, base_name, comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    ws2name = f"{base_name}.weight_scale_2"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for cluster {base_name}")
        return False
    if ws2name not in name_map:
        print(f"TensorClusters: missing .weight_scale_2 for cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    weight_scale_2 = source.tensors[name_map[ws2name]]
    groups.fp4_clusters.append(Fp4Cluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        weight_scale_2=weight_scale_2, comfy_quant=comfy_quant,
        rows=weight.dims[0], cols=weight.dims[1] * 2))
    return True


def _append_float8_cluster(source, name_map, base_name, comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    isname = f"{base_name}.input_scale"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for fp8 cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for fp8 cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    input_scale = source.tensors[name_map[isname]] if isname in name_map else None
    groups.float8_clusters.append(Float8Cluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        input_scale=input_scale, comfy_quant=comfy_quant,
        rows=weight.dims[0], cols=weight.dims[1]))
    return True


def _append_mxfp4_cluster(source, name_map, base_name, comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for mxfp4 cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for mxfp4 cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    groups.mxfp4_clusters.append(Mxfp4Cluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        comfy_quant=comfy_quant, rows=weight.dims[0], cols=weight.dims[1] * 2))
    return True


def _append_mxfp8_cluster(source, name_map, base_name, comfy_quant, groups):
    wname = f"{base_name}.weight"
    wsname = f"{base_name}.weight_scale"
    if wname not in name_map:
        print(f"TensorClusters: missing .weight for mxfp8 cluster {base_name}")
        return False
    if wsname not in name_map:
        print(f"TensorClusters: missing .weight_scale for mxfp8 cluster {base_name}")
        return False
    weight = source.tensors[name_map[wname]]
    weight_scale = source.tensors[name_map[wsname]]
    groups.mxfp8_clusters.append(Mxfp8Cluster(
        base_name=base_name, weight=weight, weight_scale=weight_scale,
        comfy_quant=comfy_quant, rows=weight.dims[0], cols=weight.dims[1]))
    return True


def _resolve_cluster_base(source, name_map, layer_key):
    wsuffix = f"{layer_key}.weight"
    if wsuffix in name_map:
        name = source.tensors[name_map[wsuffix]].name
        return name[: -len(".weight")]
    match = None
    for t in source.tensors:
        if name_suffix_match(t.name, wsuffix):
            if match is not None:
                print(f"TensorClusters: _quantization_metadata layer {layer_key} "
                      f"suffix-matches multiple .weight tensors; skipping (ambiguous)")
                return None
            match = t.name[: -len(".weight")]
    return match


def _group_from_quant_metadata(source, name_map, qm_json, groups, seen):
    try:
        parsed = json.loads(qm_json)
    except json.JSONDecodeError:
        print("TensorClusters: failed to parse _quantization_metadata")
        return
    if not isinstance(parsed, dict):
        return
    layers = parsed.get("layers")
    if not isinstance(layers, dict):
        return
    for layer_key, layer_val in layers.items():
        if not isinstance(layer_val, dict):
            continue
        layer_json = json.dumps(layer_val, separators=(",", ":"))
        scheme = parse_comfy_quant_scheme(layer_json.encode())
        if scheme == "unknown":
            print(f"TensorClusters: _quantization_metadata layer {layer_key} has an "
                  f"unsupported scheme; skipping")
            continue
        base_name = _resolve_cluster_base(source, name_map, layer_key)
        if base_name is None:
            print(f"TensorClusters: _quantization_metadata layer {layer_key} has no "
                  f"matching .weight tensor; skipping")
            continue
        if base_name in seen:
            continue
        if scheme == "nvfp4":
            _append_nvfp4_cluster(source, name_map, base_name, None, groups)
        elif scheme == "float8_e4m3fn":
            _append_float8_cluster(source, name_map, base_name, None, groups)
        elif scheme == "mxfp4":
            _append_mxfp4_cluster(source, name_map, base_name, None, groups)
        elif scheme == "mxfp8_e4m3fn":
            _append_mxfp8_cluster(source, name_map, base_name, None, groups)
        elif scheme == "int8_convrot":
            convrot = _comfy_quant_bool(layer_json.encode(), "convrot", False)
            group_size = _comfy_quant_int(layer_json.encode(), "convrot_groupsize", 256)
            _append_int8_convrot_cluster(source, name_map, base_name, convrot,
                                         group_size, None, groups)
        elif scheme == "convrot_w4a4":
            group_size = _comfy_quant_int(layer_json.encode(), "convrot_groupsize", 256)
            _append_int4_cluster(source, name_map, base_name, True, group_size, None, groups)
        elif scheme == "asym_w4a8_int8":
            group_size = _comfy_quant_int(layer_json.encode(), "group_size", ASYM_W4A8_GROUP_SIZE)
            convrot_gs = _comfy_quant_int(layer_json.encode(), "convrot_groupsize",
                                          ASYM_W4A8_CONVROT_GROUP_SIZE)
            _append_asym_w4a8_cluster(source, name_map, base_name, group_size,
                                      convrot_gs, None, groups)


def group_clusters(source, arena_alloc=None, allocator=None) -> GroupResult:
    groups = GroupResult()
    name_map = {t.name: i for i, t in enumerate(source.tensors)}
    seen_bases = set()

    for t in source.tensors:
        if not t.name.endswith(COMFY_SUFFIX):
            continue
        data = _read_tensor_bytes(source, t)
        scheme = parse_comfy_quant_scheme(data)
        base_name = t.name[: -len(COMFY_SUFFIX)]
        grouped = False
        if scheme == "nvfp4":
            grouped = _append_nvfp4_cluster(source, name_map, base_name, t, groups)
        elif scheme == "float8_e4m3fn":
            grouped = _append_float8_cluster(source, name_map, base_name, t, groups)
        elif scheme == "mxfp4":
            grouped = _append_mxfp4_cluster(source, name_map, base_name, t, groups)
        elif scheme == "mxfp8_e4m3fn":
            grouped = _append_mxfp8_cluster(source, name_map, base_name, t, groups)
        elif scheme == "int8_convrot":
            convrot = _comfy_quant_bool(data, "convrot", False)
            group_size = _comfy_quant_int(data, "convrot_groupsize", 256)
            grouped = _append_int8_convrot_cluster(source, name_map, base_name,
                                                   convrot, group_size, t, groups)
        elif scheme == "convrot_w4a4":
            group_size = _comfy_quant_int(data, "convrot_groupsize", 256)
            grouped = _append_int4_cluster(source, name_map, base_name, True,
                                           group_size, t, groups)
        elif scheme == "asym_w4a8_int8":
            group_size = _comfy_quant_int(data, "group_size", ASYM_W4A8_GROUP_SIZE)
            convrot_gs = _comfy_quant_int(data, "convrot_groupsize",
                                          ASYM_W4A8_CONVROT_GROUP_SIZE)
            grouped = _append_asym_w4a8_cluster(source, name_map, base_name,
                                                group_size, convrot_gs, t, groups)
        if grouped:
            seen_bases.add(base_name)

    meta = source.get_source_metadata()
    if meta:
        qm = meta.get("_quantization_metadata")
        if isinstance(qm, str):
            _group_from_quant_metadata(source, name_map, qm, groups, seen_bases)

    print(f"TensorClusters: found {len(groups.fp4_clusters)} nvfp4, "
          f"{len(groups.float8_clusters)} fp8, {len(groups.mxfp4_clusters)} mxfp4, "
          f"{len(groups.mxfp8_clusters)} mxfp8, {len(groups.int8_convrot_clusters)} "
          f"int8_convrot, {len(groups.int4_clusters)} int4, "
          f"{len(groups.asym_w4a8_clusters)} asym_w4a8 clusters")
    return groups


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

def _dequantize_fp4_raw(weight_bytes, scale_bytes, global_scale, rows, cols) -> np.ndarray:
    out = np.empty(rows * cols, dtype=np.float32)
    n_col_blocks = (cols // 16 + 3) // 4
    weight = np.frombuffer(weight_bytes, dtype=np.uint8)
    scale = np.frombuffer(scale_bytes, dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            byte = int(weight[row * (cols // 2) + col // 2])
            nibble = (byte >> 4) if col % 2 == 0 else (byte & 0xF)
            fp4_val = dt.LUT_FP4_E2M1[nibble]
            scale_col = col // 16
            r0 = row // 128
            r1 = row % 128
            c0 = scale_col // 4
            c1 = scale_col % 4
            scale_idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 // 32) * 4 + c1
            block_scale = dt.LUT_E4M3[scale[scale_idx]]
            out[row * cols + col] = fp4_val * block_scale * global_scale
    return out


def dequantize_fp4_cluster(cluster, source) -> np.ndarray:
    if cluster.cols % 64 != 0 or cluster.rows % 128 != 0:
        raise ValueError("InvalidClusterShape")
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_bytes = _read_tensor_bytes(source, cluster.weight_scale)
    gs_bytes = _read_tensor_bytes(source, cluster.weight_scale_2)
    global_scale = struct.unpack("<f", gs_bytes[0:4])[0]
    return _dequantize_fp4_raw(weight_bytes, scale_bytes, global_scale,
                               cluster.rows, cluster.cols)


def dequantize_float8_cluster(cluster, source) -> np.ndarray:
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_buf = _read_tensor_bytes(source, cluster.weight_scale)
    scalar_scale = struct.unpack("<f", scale_buf[0:4])[0]
    arr = np.frombuffer(weight_bytes, dtype=np.uint8)
    return (dt.LUT_E4M3[arr].astype(np.float32) * scalar_scale).astype(np.float32)


def dequantize_mxfp4_cluster(cluster, source) -> np.ndarray:
    if cluster.cols % 32 != 0:
        raise ValueError("InvalidClusterShape")
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_bytes = _read_tensor_bytes(source, cluster.weight_scale)
    rows, cols = cluster.rows, cluster.cols
    num_scale_cols = cols // 32
    weight = np.frombuffer(weight_bytes, dtype=np.uint8)
    scale = np.frombuffer(scale_bytes, dtype=np.uint8)
    out = np.empty(rows * cols, dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            byte = int(weight[row * (cols // 2) + col // 2])
            nibble = (byte & 0xF) if col % 2 == 0 else (byte >> 4)
            scale_idx = row * num_scale_cols + col // 32
            sc = float(dt.e8m0_to_f32(int(scale[scale_idx])))
            out[row * cols + col] = dt.LUT_FP4_E2M1[nibble] * sc
    return out


def dequantize_mxfp8_raw(weight_bytes, scale_bytes, rows, cols) -> np.ndarray:
    weight = np.frombuffer(weight_bytes, dtype=np.uint8)
    scale = np.frombuffer(scale_bytes, dtype=np.uint8)
    num_scale_cols = cols // 32
    flat_idx = np.arange(rows * cols)
    row = flat_idx // cols
    col = flat_idx % cols
    scale_idx = row * num_scale_cols + col // 32
    sc = np.array([float(dt.e8m0_to_f32(int(s))) for s in scale], dtype=np.float32)
    out = dt.LUT_E4M3[weight].astype(np.float32) * sc[scale_idx]
    return out.astype(np.float32)


def dequantize_mxfp8_cluster(cluster, source) -> np.ndarray:
    if cluster.cols % 32 != 0:
        raise ValueError("InvalidClusterShape")
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_bytes = _read_tensor_bytes(source, cluster.weight_scale)
    return dequantize_mxfp8_raw(weight_bytes, scale_bytes, cluster.rows, cluster.cols)


def _sign_extend_nibble(nibble: int) -> int:
    return nibble - 16 if nibble >= 8 else nibble


def dequantize_int8_convrot_raw(weight_bytes, scale_f32, rows, cols, convrot,
                                group_size) -> np.ndarray:
    if len(weight_bytes) != rows * cols or len(scale_f32) != rows:
        raise ValueError("InvalidClusterShape")
    weight = np.frombuffer(weight_bytes, dtype=np.int8).astype(np.float32)
    out = weight * scale_f32[:, None].astype(np.float32)
    out = out.ravel()
    if convrot:
        dt.rotate_groupwise_in_place(out, rows, cols, group_size)
    return out.astype(np.float32)


def dequantize_int8_convrot_cluster(cluster, source) -> np.ndarray:
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_raw = _read_tensor_bytes(source, cluster.weight_scale)
    scale_f32 = np.frombuffer(scale_raw, dtype=np.float32)
    return dequantize_int8_convrot_raw(weight_bytes, scale_f32, cluster.rows,
                                       cluster.cols, cluster.convrot,
                                       cluster.group_size)


def dequantize_int4_raw(weight_bytes, scale_f32, rows, cols, convrot,
                        group_size) -> np.ndarray:
    if cols % 2 != 0:
        raise ValueError("InvalidClusterShape")
    packed_cols = cols // 2
    if len(weight_bytes) != rows * packed_cols or len(scale_f32) != rows:
        raise ValueError("InvalidClusterShape")
    weight = np.frombuffer(weight_bytes, dtype=np.uint8)
    out = np.empty(rows * cols, dtype=np.float32)
    for row in range(rows):
        s = float(scale_f32[row])
        for pc in range(packed_cols):
            byte = int(weight[row * packed_cols + pc])
            lo = _sign_extend_nibble(byte & 0x0F)
            hi = _sign_extend_nibble(byte >> 4)
            out[row * cols + 2 * pc] = lo * s
            out[row * cols + 2 * pc + 1] = hi * s
    if convrot:
        dt.rotate_groupwise_in_place(out, rows, cols, group_size)
    return out


def dequantize_int4_cluster(cluster, source) -> np.ndarray:
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    scale_raw = _read_tensor_bytes(source, cluster.weight_scale)
    scale_f32 = np.frombuffer(scale_raw, dtype=np.float32)
    return dequantize_int4_raw(weight_bytes, scale_f32, cluster.rows,
                               cluster.cols, cluster.convrot, cluster.group_size)


def dequantize_asym_w4a8_raw(weight_bytes, s_rel_bytes, s_channel, codebook,
                             rows, cols, group_size, convrot_group_size) -> np.ndarray:
    if cols % 2 != 0 or group_size == 0 or cols % group_size != 0:
        raise ValueError("InvalidClusterShape")
    packed_cols = cols // 2
    n_groups = cols // group_size
    if len(weight_bytes) != rows * packed_cols or len(s_rel_bytes) != rows * n_groups:
        raise ValueError("InvalidClusterShape")
    if len(s_channel) != rows or len(codebook) != 16:
        raise ValueError("InvalidClusterShape")
    weight = np.frombuffer(weight_bytes, dtype=np.uint8)
    s_rel_arr = np.frombuffer(s_rel_bytes, dtype=np.uint8)
    out = np.empty(rows * cols, dtype=np.float32)
    for row in range(rows):
        sc = float(s_channel[row])
        for g in range(n_groups):
            s_rel_val = dt.fp8_e4m3_to_f32(int(s_rel_arr[row * n_groups + g]))
            levels = np.clip(dt.round_half_to_even(codebook * s_rel_val), -127.0, 127.0) * sc
            for j in range(group_size):
                col = g * group_size + j
                byte = int(weight[row * packed_cols + col // 2])
                idx = (byte & 0x0F) if col % 2 == 0 else (byte >> 4)
                out[row * cols + col] = levels[idx]
    if convrot_group_size != 0:
        dt.rotate_groupwise_in_place(out, rows, cols, convrot_group_size)
    return out


def dequantize_asym_w4a8_cluster(cluster, source) -> np.ndarray:
    weight_bytes = _read_tensor_bytes(source, cluster.weight)
    s_rel_bytes = _read_tensor_bytes(source, cluster.weight_s_rel)
    s_channel_raw = _read_tensor_bytes(source, cluster.weight_s_channel)
    codebook_raw = _read_tensor_bytes(source, cluster.weight_codebook)
    s_channel = np.frombuffer(s_channel_raw, dtype=np.float32)
    codebook = np.frombuffer(codebook_raw, dtype=np.float32)
    return dequantize_asym_w4a8_raw(weight_bytes, s_rel_bytes, s_channel,
                                    codebook, cluster.rows, cluster.cols,
                                    cluster.group_size, cluster.convrot_group_size)


def try_dequant_cluster(dest_tensor, source, groups) -> Optional[np.ndarray]:
    for cluster in groups.fp4_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_fp4_cluster(cluster, source)
    for cluster in groups.float8_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_float8_cluster(cluster, source)
    for cluster in groups.mxfp4_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_mxfp4_cluster(cluster, source)
    for cluster in groups.mxfp8_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_mxfp8_cluster(cluster, source)
    for cluster in groups.int8_convrot_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_int8_convrot_cluster(cluster, source)
    for cluster in groups.int4_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_int4_cluster(cluster, source)
    for cluster in groups.asym_w4a8_clusters:
        if name_suffix_match(cluster.weight.name, dest_tensor.name):
            return dequantize_asym_w4a8_cluster(cluster, source)
    return None


def load_matching_source_as_f32(source, name, n_elements):
    idx = best_source_match(source.tensors, name)
    if idx is None:
        return None
    source_tensor = source.tensors[idx]
    source_dtype = types_mod.from_string(source_tensor.type)
    source_size = types_mod.calc_size_in_bytes(source_dtype, n_elements)
    src_bytes = _read_tensor_bytes(source, source_tensor, source_size)
    f32_bytes = dt.convert_tensor_data(src_bytes, source_dtype, "F32", n_elements)
    return np.frombuffer(f32_bytes, dtype=np.float32), source_tensor.type


def dequant_source_to_f32(t, source, groups) -> Optional[np.ndarray]:
    cluster_f32 = try_dequant_cluster(t, source, groups)
    if cluster_f32 is not None:
        return cluster_f32
    n_elements = 1
    for d in t.dims:
        n_elements *= d
    res = load_matching_source_as_f32(source, t.name, n_elements)
    if res is not None:
        return res[0]
    return None


# ---------------------------------------------------------------------------
# Collapse
# ---------------------------------------------------------------------------

def collapse_model_tensors(model_tensors, groups, mode: str):
    if not any([
        groups.fp4_clusters, groups.float8_clusters, groups.mxfp4_clusters,
        groups.mxfp8_clusters, groups.int8_convrot_clusters,
        groups.int4_clusters, groups.asym_w4a8_clusters]):
        return
    new_tensors = []
    for t in model_tensors:
        handled = False
        for cluster in groups.fp4_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.dims = [cluster.rows, cluster.cols]
                new_t.type = "NVFP4" if mode == "preserve_quant" else "BF16"
                new_t.size = cluster.rows * cluster.cols * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or name_suffix_match(cluster.weight_scale_2.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.float8_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.type = "SCALED_F8_E4M3" if mode == "preserve_quant" else "BF16"
                n = 1
                for d in t.dims:
                    n *= d
                new_t.size = n * 2
                new_tensors.append(new_t)
                handled = True
                break
            input_scale_match = (cluster.input_scale is not None
                                 and name_suffix_match(cluster.input_scale.name, t.name))
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))
                    or input_scale_match):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.mxfp4_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.dims = [cluster.rows, cluster.cols]
                new_t.type = "MXFP4" if mode == "preserve_quant" else "BF16"
                new_t.size = cluster.rows * cluster.cols * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.mxfp8_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.type = "MXFP8_E4M3" if mode == "preserve_quant" else "BF16"
                n = 1
                for d in t.dims:
                    n *= d
                new_t.size = n * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.int8_convrot_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.dims = [cluster.rows, cluster.cols]
                new_t.type = ("INT8_CONVROT" if cluster.convrot else "INT8") if mode == "preserve_quant" else "BF16"
                new_t.size = cluster.rows * cluster.cols * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.int4_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.dims = [cluster.rows, cluster.cols]
                new_t.type = "INT4_CONVROT" if mode == "preserve_quant" else "BF16"
                new_t.size = cluster.rows * cluster.cols * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_scale.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if handled:
            continue
        for cluster in groups.asym_w4a8_clusters:
            if name_suffix_match(cluster.weight.name, t.name):
                new_t = t.dupe()
                new_t.dims = [cluster.rows, cluster.cols]
                new_t.type = "ASYM_W4A8_INT8" if mode == "preserve_quant" else "BF16"
                new_t.size = cluster.rows * cluster.cols * 2
                new_tensors.append(new_t)
                handled = True
                break
            if (name_suffix_match(cluster.weight_s_rel.name, t.name)
                    or name_suffix_match(cluster.weight_s_channel.name, t.name)
                    or name_suffix_match(cluster.weight_codebook.name, t.name)
                    or (cluster.comfy_quant is not None and name_suffix_match(cluster.comfy_quant.name, t.name))):
                handled = True
                break
        if not handled:
            new_tensors.append(t)
    model_tensors[:] = new_tensors


# ---------------------------------------------------------------------------
# Write path: cluster layout + encoding
# ---------------------------------------------------------------------------

def is_cluster_type(dtype: str) -> bool:
    return dtype in ("SCALED_F8_E4M3", "MXFP4", "MXFP8_E4M3", "NVFP4",
                     "INT8", "INT8_CONVROT", "INT4_CONVROT",
                     "INT4_CONVROT_SR", "ASYM_W4A8_INT8")


def _rows_cols(dims):
    cols = dims[-1] if len(dims) >= 1 else 0
    rows = 1
    if len(dims) >= 2:
        for d in dims[:len(dims) - 1]:
            rows *= d
    return rows, cols


class SubTensorSpec:
    def __init__(self, suffix: str, dtype: str, dims, bytes_: int):
        self.suffix = suffix
        self.dtype = dtype
        self.dims = list(dims)
        self.bytes = bytes_


def cluster_write_layout(dtype: str, dims) -> Optional[List[SubTensorSpec]]:
    rows, cols = _rows_cols(dims)
    n_elements = rows * cols
    if dtype == "SCALED_F8_E4M3":
        return [
            SubTensorSpec(".weight", "F8_E4M3", dims, n_elements),
            SubTensorSpec(".weight_scale", "F32", [], 4),
            SubTensorSpec(".comfy_quant", "U8", [len(FP8_COMFY_JSON)], len(FP8_COMFY_JSON)),
        ]
    if dtype == "MXFP4":
        return [
            SubTensorSpec(".weight", "U32", [rows, cols // 8], rows * cols // 2),
            SubTensorSpec(".weight_scale", "U8", [rows, (cols + 31) // 32], rows * ((cols + 31) // 32)),
            SubTensorSpec(".comfy_quant", "U8", [len(MXFP4_COMFY_JSON)], len(MXFP4_COMFY_JSON)),
        ]
    if dtype == "MXFP8_E4M3":
        nsc = (cols + 31) // 32
        nrb = (rows + 127) // 128
        ncb = (nsc + 3) // 4
        return [
            SubTensorSpec(".weight", "F8_E4M3", dims, n_elements),
            SubTensorSpec(".weight_scale", "U8", [nrb * 128, ncb * 4], nrb * 128 * ncb * 4),
            SubTensorSpec(".comfy_quant", "U8", [len(MXFP8_COMFY_JSON)], len(MXFP8_COMFY_JSON)),
        ]
    if dtype == "NVFP4":
        return [
            SubTensorSpec(".weight", "U8", [rows, cols // 2], rows * (cols // 2)),
            SubTensorSpec(".weight_scale", "F8_E4M3", [rows, cols // 16], rows * (cols // 16)),
            SubTensorSpec(".weight_scale_2", "F32", [], 4),
            SubTensorSpec(".comfy_quant", "U8", [len(NVFP4_COMFY_JSON)], len(NVFP4_COMFY_JSON)),
        ]
    if dtype in ("INT8", "INT8_CONVROT"):
        comfy_json = INT8_CONVROT_COMFY_JSON if dtype == "INT8_CONVROT" else INT8_COMFY_JSON
        return [
            SubTensorSpec(".weight", "I8", dims, n_elements),
            SubTensorSpec(".weight_scale", "F32", [rows, 1], rows * 4),
            SubTensorSpec(".comfy_quant", "U8", [len(comfy_json)], len(comfy_json)),
        ]
    if dtype in ("INT4_CONVROT", "INT4_CONVROT_SR"):
        return [
            SubTensorSpec(".weight", "I8", [rows, cols // 2], rows * (cols // 2)),
            SubTensorSpec(".weight_scale", "F32", [rows], rows * 4),
            SubTensorSpec(".comfy_quant", "U8", [len(INT4_CONVROT_COMFY_JSON)], len(INT4_CONVROT_COMFY_JSON)),
        ]
    if dtype == "ASYM_W4A8_INT8":
        n_groups = cols // ASYM_W4A8_GROUP_SIZE
        return [
            SubTensorSpec(".weight", "I8", [rows, cols // 2], rows * (cols // 2)),
            SubTensorSpec(".weight_s_rel", "F8_E4M3", [rows, n_groups], rows * n_groups),
            SubTensorSpec(".weight_s_channel", "F32", [rows], rows * 4),
            SubTensorSpec(".weight_codebook", "F32", [16], 16 * 4),
            SubTensorSpec(".comfy_quant", "U8", [len(ASYM_W4A8_COMFY_JSON)], len(ASYM_W4A8_COMFY_JSON)),
        ]
    return None


def cluster_write_size(dtype: str, dims) -> Optional[int]:
    specs = cluster_write_layout(dtype, dims)
    if specs is None:
        return None
    return sum(s.bytes for s in specs)


def _quantize_to_nvfp4_raw(data: np.ndarray, rows, cols):
    if cols % 64 != 0 or rows % 128 != 0:
        raise ValueError("InvalidClusterShape")
    data = np.ascontiguousarray(data, dtype=np.float32)
    num_scale_cols = cols // 16
    n_col_blocks = (num_scale_cols + 3) // 4
    max_abs = float(np.max(np.abs(data))) if data.size else 0.0
    global_scale = np.float32(max_abs / (6.0 * 448.0)) if max_abs > 0.0 else np.float32(1.0)
    inv_global = 1.0 / global_scale
    scale = np.zeros(rows * num_scale_cols, dtype=np.uint8)
    weight = np.zeros(rows * (cols // 2), dtype=np.uint8)
    mat = data.reshape(rows, cols)
    for row in range(rows):
        r0 = row // 128
        r1 = row % 128
        for sc_idx in range(num_scale_cols):
            block = mat[row, sc_idx * 16:(sc_idx + 1) * 16] * inv_global
            block_max = float(np.max(np.abs(block))) if block.size else 0.0
            if block_max > 0.0:
                scale_byte = int(np.asarray(dt.f32_to_fp8_e4m3(block_max / 6.0)).reshape(-1)[0])
            else:
                scale_byte = 0
            c0 = sc_idx // 4
            c1 = sc_idx % 4
            idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 // 32) * 4 + c1
            scale[idx] = scale_byte
        for col in range(cols):
            sc_idx = col // 16
            c0 = sc_idx // 4
            c1 = sc_idx % 4
            scale_idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 // 32) * 4 + c1
            block_scale = float(dt.LUT_E4M3[scale[scale_idx]])
            val = float(mat[row, col]) * float(inv_global)
            nibble = 0 if block_scale == 0.0 else int(np.asarray(dt.f32_to_fp4_e2m1(val / block_scale)).reshape(-1)[0])
            byte_idx = row * (cols // 2) + col // 2
            if col % 2 == 0:
                weight[byte_idx] |= (nibble & 0xF) << 4
            else:
                weight[byte_idx] |= nibble & 0xF
    return weight, scale, global_scale


def write_cluster_data(writer, dtype: str, f32_data: np.ndarray, dims,
                       stochastic_rounding: int = DEFAULT_STOCHASTIC_SEED):
    rows, cols = _rows_cols(dims)
    f32_data = np.ascontiguousarray(f32_data, dtype=np.float32)
    if dtype == "SCALED_F8_E4M3":
        weight, scale = dt.quantize_to_comfy_fp8(f32_data)
        writer.write(weight.tobytes())
        writer.write(struct.pack("<f", scale))
        writer.write(FP8_COMFY_JSON)
    elif dtype == "MXFP4":
        weight, scale = dt.quantize_to_comfy_mxfp4(f32_data)
        writer.write(weight.tobytes())
        writer.write(scale.tobytes())
        writer.write(MXFP4_COMFY_JSON)
    elif dtype == "MXFP8_E4M3":
        weight, scale = dt.quantize_to_comfy_mxfp8(f32_data)
        n_scale_cols = (cols + 31) // 32
        blocked_scale = dt.to_blocked_mxfp8(scale, rows, n_scale_cols)
        writer.write(weight.tobytes())
        writer.write(blocked_scale.tobytes())
        writer.write(MXFP8_COMFY_JSON)
    elif dtype == "NVFP4":
        weight, scale, global_scale = _quantize_to_nvfp4_raw(f32_data, rows, cols)
        writer.write(weight.tobytes())
        writer.write(scale.tobytes())
        writer.write(struct.pack("<f", global_scale))
        writer.write(NVFP4_COMFY_JSON)
    elif dtype in ("INT8", "INT8_CONVROT"):
        is_convrot = dtype == "INT8_CONVROT"
        weight, scale = dt.quantize_to_int8(f32_data, rows, cols, is_convrot,
                                            INT8_CONVROT_GROUP_SIZE)
        writer.write(weight.tobytes())
        writer.write(scale.astype(np.float32).tobytes())
        writer.write(INT8_CONVROT_COMFY_JSON if is_convrot else INT8_COMFY_JSON)
    elif dtype in ("INT4_CONVROT", "INT4_CONVROT_SR"):
        seed = stochastic_rounding if dtype == "INT4_CONVROT_SR" else 0
        weight, scale = dt.quantize_to_int4(f32_data, rows, cols, True,
                                            INT4_CONVROT_GROUP_SIZE, seed)
        writer.write(weight.tobytes())
        writer.write(scale.astype(np.float32).tobytes())
        writer.write(INT4_CONVROT_COMFY_JSON)
    elif dtype == "ASYM_W4A8_INT8":
        weight, s_rel, s_channel, recommended = dt.quantize_to_asym_w4a8(
            f32_data, rows, cols, ASYM_W4A8_CONVROT_GROUP_SIZE)
        if recommended:
            print("ASYM_W4A8_INT8: rotated weights are heavy-tailed; comfy_kitchen "
                  "would fit a per-tensor codebook here, ggufy writes the frozen "
                  "Lloyd-Max table")
        writer.write(weight.tobytes())
        writer.write(s_rel.tobytes())
        writer.write(s_channel.astype(np.float32).tobytes())
        writer.write(dt.W4A8_CODEBOOK.astype(np.float32).tobytes())
        writer.write(ASYM_W4A8_COMFY_JSON)
    else:
        raise ValueError("NotAClusterType")
