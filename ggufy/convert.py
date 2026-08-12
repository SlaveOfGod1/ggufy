"""Conversion pipeline: SafeTensors/GGUF -> GGUF/SafeTensors.

Port of src/Convert.zig: architecture detection, tensor filtering / prefix
stripping, cluster collapse, quantization-type assignment (template or auto,
with sensitivity-aware quantization), shape fix, layout assignment, metadata
construction and the output writers.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import data_transform as dt
from . import ggml as ggml_mod
from . import image_arch as arch_mod
from . import safetensor as st_mod
from . import tensor_clusters as tc
from . import types as types_mod
from .gguf import Gguf, calculate_file_size as gguf_calc_file_size
from .safetensor import calculate_file_size as st_calc_file_size
from .types import FileType, Tensor

QUANTIZATION_THRESHOLD = 256 * 256
REARRANGE_THRESHOLD = 512

_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"]


def format_bytes(bytes_: int) -> str:
    value = float(bytes_)
    unit = 0
    while value >= 1024.0 and unit < len(_UNITS) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.2f} {_UNITS[unit]}"


SOURCE_QUANT_METADATA_KEY = "_quantization_metadata"
GGUFY_REPO_URL = "https://github.com/qskousen/ggufy"
GGUFY_VERSION = "0.1.0"


class QuantizationFamilies:
    def __init__(self, allow_0=False, allow_1=False, allow_k=False):
        self.allow_0 = allow_0
        self.allow_1 = allow_1
        self.allow_k = allow_k

    @staticmethod
    def parse(s: str) -> "QuantizationFamilies":
        result = QuantizationFamilies()
        for tok in s.split(","):
            trimmed = tok.strip()
            if trimmed == "0":
                result.allow_0 = True
            elif trimmed == "1":
                result.allow_1 = True
            elif trimmed == "k":
                result.allow_k = True
            else:
                raise ValueError("InvalidQuantizationFamily")
        if not (result.allow_0 or result.allow_1 or result.allow_k):
            raise ValueError("InvalidQuantizationFamily")
        return result

    @staticmethod
    def from_data_type(dtype: str) -> "QuantizationFamilies":
        return QuantizationFamilies(
            allow_0=dtype.endswith("_0"),
            allow_1=dtype.endswith("_1"),
            allow_k=dtype.endswith("_k"),
        )

    def allows(self, level: str) -> bool:
        if level in ("q8_0", "f16", "bf16", "f32", "f64"):
            return True
        if level in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k"):
            return self.allow_k
        if level in ("q4_0", "q5_0"):
            return self.allow_0
        if level in ("q4_1", "q5_1"):
            return self.allow_1
        return False


QUANTIZATION_LEVELS = [
    "q2_k", "q3_k", "q4_0", "q4_1", "q4_k", "q5_0", "q5_1", "q5_k",
    "q6_k", "q8_0", "f16", "bf16", "f32", "f64",
]
QUANT_LEVEL_INDEX = {name: i for i, name in enumerate(QUANTIZATION_LEVELS)}


def quantization_level_from_string(s: str) -> str:
    lower = s.lower()
    if lower not in QUANT_LEVEL_INDEX:
        raise ValueError(f"UnknownQuantizationType: {s}")
    return lower


def detect_upscaling(source_tensors: List[Tensor], target_dtype: Optional[str]) -> bool:
    if target_dtype is None:
        return False
    target_rank = types_mod.precision_rank(target_dtype)
    if target_rank == 255:
        return False
    for t in source_tensors:
        src_rank = types_mod.precision_rank(t.type)
        if src_rank != 255 and target_rank > src_rank:
            return True
    return False


def datatype_fits_filetype(datatype: Optional[str], filetype: str) -> bool:
    if datatype is None:
        return True
    try:
        types_mod.for_format(datatype, filetype)
        return True
    except ValueError:
        return False


def validate_datatype_for_filetype(datatype: Optional[str], filetype: str):
    if datatype is None:
        return
    if not datatype_fits_filetype(datatype, filetype):
        if filetype == FileType.SAFETENSORS:
            print(f"ERROR: {datatype} is a GGUF block-quantized type and cannot be "
                  f"stored in a SafeTensors file. Use -f gguf to write a GGUF, or "
                  f"choose a SafeTensors type (F16, BF16, F8_E4M3, SCALED_F8_E4M3, "
                  f"INT8, INT8_CONVROT, INT4_CONVROT, ASYM_W4A8_INT8, MXFP4, "
                  f"MXFP8_E4M3, NVFP4).")
        else:
            print(f"ERROR: {datatype} has no GGUF representation and can only be "
                  f"written to a SafeTensors file. Use -f safetensors, or choose a "
                  f"GGUF type (f16, bf16, f32, q8_0, q6_k, q5_k, q4_k, q3_k, q2_k, ...).")
        raise ValueError("DatatypeNotRepresentableInFiletype")


def compute_output_path(opts: "ConvertOptions") -> str:
    dir_path = opts.output_dir if opts.output_dir else (os.path.dirname(opts.path) or ".")
    ext = "gguf" if opts.filetype == FileType.GGUF else "safetensors"
    if opts.output_name:
        base_name = opts.output_name
    else:
        stem = os.path.splitext(os.path.basename(opts.path))[0]
        dtype_str = opts.datatype if opts.datatype else ("f16" if opts.filetype == FileType.GGUF else "F16")
        base_name = f"{stem}-{dtype_str}"
    return os.path.join(dir_path, f"{base_name}.{ext}")


@dataclass
class PreparedConversion:
    arch: Any
    model_tensors: List[Tensor]
    template_metadata: Optional[Dict[str, Any]]
    extra_metadata: Dict[str, Any]
    groups: tc.GroupResult


def prepare_conversion(f, opts: "ConvertOptions"):
    arch = None
    try:
        arch = arch_mod.detect_arch_from_tensors_or_error(f.tensors)
    except ValueError:
        if opts.allow_unknown_arch:
            print("WARNING: Unknown architecture; proceeding anyway. Results may be suboptimal.")
            arch = arch_mod.GENERIC_ARCH
        else:
            raise
    threshold = arch.threshhold if arch.threshhold is not None else QUANTIZATION_THRESHOLD
    print(f"Detected architecture: {arch.name}")

    model_tensors = filter_and_strip_tensors(f, arch, opts.filetype, opts.model_only)

    restore_orig_shapes(model_tensors, f.get_source_metadata())

    groups = tc.group_clusters(f)
    tc.collapse_model_tensors(model_tensors, groups, "dequant")

    template_metadata = None
    if opts.template_path:
        template_metadata = apply_template(opts.template_path, model_tensors, opts.filetype)
    else:
        assign_quant_types(model_tensors, arch, threshold, opts)

    extra_metadata: Dict[str, Any] = {}
    if arch.shape_fix and opts.filetype == FileType.GGUF:
        apply_shape_fix(model_tensors, extra_metadata)

    if opts.filetype == FileType.GGUF:
        model_tensors.sort(key=lambda t: t.name)

    assign_output_layout(model_tensors, opts)

    return PreparedConversion(arch=arch, model_tensors=model_tensors,
                              template_metadata=template_metadata,
                              extra_metadata=extra_metadata, groups=groups)


def convert(f, opts: "ConvertOptions"):
    if not opts.allow_upscale and detect_upscaling(f.tensors, opts.datatype):
        print("ERROR: Source contains lossy-quantized tensors; converting to a higher-"
              "precision format will NOT recover lost information — the extra bits are "
              "fill-in only. Pass --allow-upscale (-U) to convert anyway.")
        raise ValueError("UpscalingNotAllowed")

    prep = prepare_conversion(f, opts)

    if opts.filetype == FileType.GGUF:
        write_gguf(f, prep, opts)
    else:
        write_safetensors(f, prep, opts)


def predict_output_size(f, opts: "ConvertOptions") -> int:
    prep = prepare_conversion(f, opts)
    if opts.filetype == FileType.GGUF:
        metadata: Dict[str, Any] = {}
        build_gguf_metadata(metadata, f, prep.arch, prep.template_metadata,
                            prep.extra_metadata, opts)
        return gguf_calc_file_size(prep.model_tensors, metadata, 32)
    else:
        metadata = build_safetensors_metadata(f, prep.template_metadata, prep.extra_metadata)
        return st_calc_file_size(prep.model_tensors, metadata)


# ---------------------------------------------------------------------------
# Quantization level helpers
# ---------------------------------------------------------------------------

def calculate_quantization_level(sensitivity: float, aggressiveness: float,
                                 target_level: str, source_type: str,
                                 families: QuantizationFamilies) -> str:
    sens = max(1.0, min(100.0, sensitivity))
    hard = max(1.0, min(100.0, aggressiveness))
    source_level = quantization_level_from_string(source_type)
    source_idx = QUANT_LEVEL_INDEX[source_level]
    target_idx = QUANT_LEVEL_INDEX[target_level]

    skip_f16 = source_idx > QUANT_LEVEL_INDEX["f16"]
    allowed = []
    for i in range(target_idx, source_idx + 1):
        candidate = QUANTIZATION_LEVELS[i]
        if skip_f16 and candidate == "f16":
            continue
        if families.allows(candidate):
            allowed.append(candidate)
    if not allowed:
        return source_level

    norm_sens = (sens - 1.0) / 99.0
    hardness_factor = hard / 100.0
    exponent = 0.5 + (hardness_factor * 3.0)
    adjusted_sens = norm_sens ** exponent
    max_idx = len(allowed) - 1
    raw = adjusted_sens * max_idx
    picked = min(int(round(raw)), max_idx)
    return allowed[picked]


# ---------------------------------------------------------------------------
# Step 1 - filter tensors and strip name prefixes
# ---------------------------------------------------------------------------

def filter_and_strip_tensors(f, arch: Arch, output_filetype: str,
                             model_only: bool) -> List[Tensor]:
    model_tensors = []

    if output_filetype == FileType.SAFETENSORS and not model_only:
        for t in f.tensors:
            if not arch.should_ignore(t.name):
                model_tensors.append(t.dupe())
        return model_tensors

    has_model_prefix = any(t.name.startswith("model.") for t in f.tensors)
    for t in f.tensors:
        if has_model_prefix:
            if t.name.startswith("model."):
                if not arch.should_ignore(t.name):
                    model_tensors.append(t.dupe())
            else:
                print(f"Filtering out tensor: {t.name}")
        else:
            if not arch.should_ignore(t.name):
                model_tensors.append(t.dupe())

    for t in model_tensors:
        t.name = arch_mod.strip_prefix(t.name)
    return model_tensors


# ---------------------------------------------------------------------------
# Step 2a - apply a JSON template
# ---------------------------------------------------------------------------

def find_source_tensor(tensors: List[Tensor], target_name: str) -> Optional[Tensor]:
    for t in tensors:
        if t.name == target_name:
            return t
        if (len(t.name) > len(target_name)
                and t.name[len(t.name) - len(target_name) - 1] == '.'
                and t.name.endswith(target_name)):
            return t
    return None


def apply_template_entry(src: Tensor, target_name: str, target_info: Dict,
                         output_filetype: str) -> Tensor:
    target_shape = [int(v) for v in target_info["shape"]]
    target_dims = [0] * len(target_shape)
    target_elements = 1
    for i, item in enumerate(target_shape):
        target_dims[len(target_shape) - 1 - i] = item
        target_elements *= item

    target_type = target_info["type"]
    source_elements = 1
    for d in src.dims:
        source_elements *= d
    if source_elements != target_elements:
        raise ValueError(f"Tensor {target_name} shape mismatch. Source elements: "
                         f"{source_elements}, Target elements: {target_elements}")

    raw_type = types_mod.from_string(target_type)
    data_type = types_mod.for_format(raw_type, output_filetype)

    if types_mod.format_type(data_type) == FileType.GGUF:
        ggml_type = ggml_mod.GgmlType.from_string(data_type)
        bs = ggml_mod.GgmlType.get_block_size(ggml_type)
        if bs > 1 and source_elements % bs != 0:
            raise ValueError(f"Tensor {target_name} cannot be quantized to type "
                             f"{data_type}. Element count {source_elements} is not "
                             f"a multiple of block size {bs}")

    new_t = src.dupe()
    new_t.name = target_name
    new_t.dims = target_dims
    new_t.type = data_type
    new_t.size = types_mod.calc_size_in_bytes(data_type, target_elements)
    return new_t


def apply_template(template_path: str, model_tensors: List[Tensor],
                   output_filetype: str) -> Optional[Dict[str, Any]]:
    print(f"Using template {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        t_json = json.load(f)

    template_metadata = t_json.get("metadata")
    t_tensors = t_json.get("tensors")
    if t_tensors is None:
        raise ValueError("InvalidTemplate")

    filtered = []
    for target_name, target_info in t_tensors.items():
        source_tensor = find_source_tensor(model_tensors, target_name)
        if source_tensor is not None:
            new_t = apply_template_entry(source_tensor, target_name, target_info, output_filetype)
            filtered.append(new_t)
            print(f"Matched target tensor {target_name} to source tensor "
                  f"{source_tensor.name}, setting to type {new_t.type}")
        else:
            print(f"WARNING: Template tensor {target_name} not found in source file.")
    model_tensors[:] = filtered
    return template_metadata


# ---------------------------------------------------------------------------
# Step 2b - auto-assign quantization types
# ---------------------------------------------------------------------------

_EMBEDDING_SUFFIXES = [
    ".embed.weight",
    ".embed_tokens.weight",
    ".token_embedding.weight",
    ".token_embed.weight",
    ".word_embeddings.weight",
    ".tok_embeddings.weight",
    ".wte.weight",
]


def is_embedding_weight(name: str) -> bool:
    return any(name.endswith(s) for s in _EMBEDDING_SUFFIXES)


def nearest_compatible_type(t: Tensor, opts: "ConvertOptions", num_elements: int):
    source_type = types_mod.from_string(t.type)
    if opts.filetype == FileType.GGUF:
        if source_type in ("F8_E4M3", "F8_E5M2"):
            t.type = "f16"
            t.size = ggml_mod.GgmlType.calc_size_in_bytes("f16", num_elements)
            return
        if source_type == "BF16":
            t.type = "f32"
            t.size = ggml_mod.GgmlType.calc_size_in_bytes("f32", num_elements)
            return


def cluster_eligible(t: Tensor, ttype: str, num_elements: int, arch: Arch) -> bool:
    if not t.name.endswith(".weight"):
        return False
    n_cols = t.dims[-1] if len(t.dims) >= 1 else 0
    if ttype == "SCALED_F8_E4M3":
        return True
    if ttype in ("MXFP4", "MXFP8_E4M3"):
        return len(t.dims) >= 1 and n_cols >= 32
    if ttype == "NVFP4":
        return (len(t.dims) >= 1 and n_cols >= 64 and n_cols % 64 == 0
                and (num_elements // n_cols) % 128 == 0
                and not arch.is_nvfp4_passthrough(t.name))
    if ttype == "INT8":
        return len(t.dims) == 2 and n_cols >= 1
    if ttype == "INT8_CONVROT":
        return len(t.dims) == 2 and n_cols % tc.INT8_CONVROT_GROUP_SIZE == 0
    if ttype in ("INT4_CONVROT", "INT4_CONVROT_SR"):
        return len(t.dims) == 2 and n_cols % tc.INT4_CONVROT_GROUP_SIZE == 0
    if ttype == "ASYM_W4A8_INT8":
        return (len(t.dims) == 2
                and n_cols % tc.ASYM_W4A8_CONVROT_GROUP_SIZE == 0
                and n_cols % tc.ASYM_W4A8_GROUP_SIZE == 0)
    return False


def assign_tensor_type(t: Tensor, num_elements: int, arch: Arch, threshold: int,
                       opts: "ConvertOptions", use_sensitivity: bool,
                       sensitivities: Optional[Dict[str, Any]]):
    if arch.should_upcast(t.name) and opts.filetype == FileType.GGUF:
        print(f"Forcing layer {t.name} to f32 for compatibility")
        t.type = "f32"
        t.size = ggml_mod.GgmlType.calc_size_in_bytes("f32", num_elements)
        return

    if opts.filetype == FileType.GGUF and len(t.dims) <= 1:
        return nearest_compatible_type(t, opts, num_elements)

    if is_embedding_weight(t.name):
        return nearest_compatible_type(t, opts, num_elements)

    if num_elements < threshold:
        return nearest_compatible_type(t, opts, num_elements)

    if arch.is_high_precision(t.name):
        return nearest_compatible_type(t, opts, num_elements)

    ttype = opts.datatype
    if ttype is None:
        return
    if opts.filetype == FileType.GGUF:
        ggml_type = ggml_mod.GgmlType.from_string(ttype)
        bs = ggml_mod.GgmlType.get_block_size(ggml_type)
        if bs > 1 and num_elements % bs != 0:
            print(f"Cannot convert tensor {t.name} to type {ttype} because "
                  f"{num_elements} is not a multiple of blocksize {bs}")
            return

    if opts.filetype == FileType.SAFETENSORS and tc.is_cluster_type(ttype):
        if cluster_eligible(t, ttype, num_elements, arch):
            t.type = ttype
            t.size = tc.cluster_write_size(ttype, t.dims)
            return
        return nearest_compatible_type(t, opts, num_elements)

    if use_sensitivity:
        apply_sensitivity_quantization(t, num_elements, ttype,
                                       opts.quantization_aggressiveness,
                                       resolved_families(opts), sensitivities)
    else:
        t.type = ttype
        t.size = types_mod.calc_size_in_bytes(ttype, num_elements)


def assign_quant_types(model_tensors: List[Tensor], arch: Arch, threshold: int,
                       opts: "ConvertOptions"):
    use_sensitivity = False
    sensitivities = None

    if not opts.skip_sensitivity and opts.filetype == FileType.GGUF:
        if opts.sensitivities_path:
            print(f"Using user-supplied sensitivities file: {opts.sensitivities_path}")
            with open(opts.sensitivities_path, "r", encoding="utf-8") as f:
                sensitivities = json.load(f)
            use_sensitivity = True
        elif len(arch.sensitivities) > 1:
            sensitivities = json.loads(arch.sensitivities)
            use_sensitivity = True

    for t in model_tensors:
        num_elements = 1
        for d in t.dims:
            num_elements *= d
        assign_tensor_type(t, num_elements, arch, threshold, opts,
                           use_sensitivity, sensitivities)

        if opts.filetype == FileType.GGUF and t.type in ("f64", "F64"):
            print(f"Downcasting unsupported f64 to f32 for tensor {t.name}")
            t.type = "f32"
            t.size = ggml_mod.GgmlType.calc_size_in_bytes("f32", num_elements)


def assign_output_layout(model_tensors: List[Tensor], opts: "ConvertOptions"):
    offset = 0
    for t in model_tensors:
        if opts.filetype == FileType.SAFETENSORS:
            try:
                dt_type = types_mod.from_string(t.type)
                if tc.is_cluster_type(dt_type):
                    cs = tc.cluster_write_size(dt_type, t.dims)
                    if cs is not None:
                        t.size = cs
            except ValueError:
                pass
        if opts.filetype == FileType.GGUF:
            padding_len = (32 - (t.size % 32)) % 32
            t.offset = offset
            offset += t.size + padding_len
        else:
            t.offset = offset
            offset += t.size


def resolved_families(opts: "ConvertOptions") -> QuantizationFamilies:
    if opts.allowed_quant_families is not None:
        return opts.allowed_quant_families
    if opts.datatype:
        derived = QuantizationFamilies.from_data_type(opts.datatype)
        if derived.allow_0 or derived.allow_1 or derived.allow_k:
            return derived
    return QuantizationFamilies(True, True, True)


def apply_sensitivity_quantization(t: Tensor, num_elements: int, dtype: str,
                                   aggressiveness: float,
                                   families: QuantizationFamilies,
                                   sensitivities: Dict[str, Any]):
    if t.name in sensitivities:
        sv = sensitivities[t.name]
        sens = float(sv)
        target_level = quantization_level_from_string(dtype)
        quant_level = calculate_quantization_level(sens, aggressiveness,
                                                   target_level, t.type, families)
        final_ggml_type = ggml_mod.GgmlType.from_string(quant_level)
        print(f"Layer {t.name}: sensitivity={sens:.1f}, hardness={aggressiveness}, "
              f"{dtype} -> {quant_level}")
        t.type = quant_level
        t.size = ggml_mod.GgmlType.calc_size_in_bytes(final_ggml_type, num_elements)
    else:
        print(f"No sensitivity data for layer {t.name}, using target type")
        t.type = dtype
        t.size = types_mod.calc_size_in_bytes(dtype, num_elements)


# ---------------------------------------------------------------------------
# Step 3 - shape fix
# ---------------------------------------------------------------------------

def restore_orig_shapes(model_tensors: List[Tensor], src_meta: Optional[Dict[str, Any]]):
    if not src_meta:
        return
    for t in model_tensors:
        key = f"comfy.gguf.orig_shape.{t.name}"
        value = src_meta.get(key)
        if not isinstance(value, list):
            continue
        dims = []
        ok = True
        for item in value:
            if isinstance(item, int):
                dims.append(item)
            else:
                ok = False
                break
        if not ok:
            continue
        t.dims = dims
        print(f"Restored original shape for {t.name}")


def apply_shape_fix(model_tensors: List[Tensor], extra_metadata: Dict[str, Any]):
    for t in model_tensors:
        n_elements = 1
        for d in t.dims:
            n_elements *= int(d)
        n_dims = len(t.dims)
        last_dim = t.dims[n_dims - 1] if n_dims > 0 else 0
        if n_dims <= 1:
            continue
        if n_elements < REARRANGE_THRESHOLD:
            continue
        if n_elements % 256 != 0:
            continue
        if last_dim % 256 == 0:
            continue
        extra_metadata[f"comfy.gguf.orig_shape.{t.name}"] = list(t.dims)
        t.dims = [n_elements // 256, 256]
        print(f"Applied shape fix to {t.name}: new shape {{ {t.dims[0]}, {t.dims[1]} }}")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def merge_base_config(metadata: Dict[str, Any], base_json: str):
    base = json.loads(base_json)
    existing_str = metadata.get("config")
    if isinstance(existing_str, str) and existing_str:
        try:
            existing = json.loads(existing_str)
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    merged = {}
    merged.update(base)
    merged.update(existing)
    metadata["config"] = json.dumps(merged)


def gguf_file_type(datatype: Optional[str]) -> int:
    if datatype is None:
        return 1
    m = {
        "f32": 0, "f16": 1, "q4_0": 2, "q4_1": 3, "q5_0": 8, "q5_1": 9,
        "q8_0": 7, "q2_k": 10, "q3_k": 12, "q4_k": 15, "q5_k": 17,
        "q6_k": 18, "bf16": 37,
    }
    return m.get(datatype, 1)


def stamp_converter_provenance(metadata: Dict[str, Any]):
    for key, value in (("converted_by", f"ggufy {GGUFY_VERSION}"),
                       ("converter_url", GGUFY_REPO_URL),
                       ("converter_note", "Converted with ggufy")):
        if key in metadata:
            metadata[key] = value


def build_gguf_metadata(metadata: Dict[str, Any], f, arch, template_metadata,
                        extra_metadata, opts: "ConvertOptions"):
    arch_name = opts.arch_override if opts.arch_override else arch.name
    metadata["general.architecture"] = arch_name
    metadata["general.quantization_version"] = 2
    metadata["general.file_type"] = gguf_file_type(opts.datatype)

    if template_metadata:
        for k, v in template_metadata.items():
            if k not in metadata:
                metadata[k] = v
    elif f.get_source_metadata():
        for k, v in f.get_source_metadata().items():
            if k not in metadata:
                metadata[k] = v

    for k, v in extra_metadata.items():
        if k not in metadata:
            metadata[k] = v

    metadata.pop(SOURCE_QUANT_METADATA_KEY, None)
    stamp_converter_provenance(metadata)

    if arch.base_config_json:
        merge_base_config(metadata, arch.base_config_json)


def build_safetensors_metadata(f, template_metadata, extra_metadata) -> Optional[Dict[str, Any]]:
    metadata = dict(f.get_source_metadata()) if f.get_source_metadata() else None
    if template_metadata:
        if metadata is None:
            metadata = {}
        for k, v in template_metadata.items():
            if k not in metadata:
                metadata[k] = v
    elif f.get_source_metadata():
        for k, v in f.get_source_metadata().items():
            if k not in metadata:
                metadata[k] = v
    for k, v in extra_metadata.items():
        if k not in metadata:
            metadata[k] = v
    if metadata is not None:
        metadata.pop(SOURCE_QUANT_METADATA_KEY, None)
        stamp_converter_provenance(metadata)
    return metadata


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _resolve_output_path(opts: "ConvertOptions") -> str:
    dir_path = opts.output_dir if opts.output_dir else (os.path.dirname(opts.path) or ".")
    os.makedirs(dir_path, exist_ok=True)
    if opts.output_name:
        base_name = opts.output_name
    else:
        stem = os.path.splitext(os.path.basename(opts.path))[0]
        dtype_str = opts.datatype if opts.datatype else ("f16" if opts.filetype == FileType.GGUF else "F16")
        base_name = f"{stem}-{dtype_str}"
    ext = "gguf" if opts.filetype == FileType.GGUF else "safetensors"
    return os.path.join(dir_path, f"{base_name}.{ext}")


def write_gguf(f, prep: PreparedConversion, opts: "ConvertOptions"):
    out_filename = _resolve_output_path(opts)
    out_gguf = Gguf(out_filename, overwrite=True)
    try:
        out_gguf.tensors = prep.model_tensors
        build_gguf_metadata(out_gguf.metadata, f, prep.arch,
                            prep.template_metadata, prep.extra_metadata, opts)
        try:
            out_gguf.save_with_st_data(f, threads=opts.threads, groups=prep.groups)
        except RuntimeError:
            try:
                os.remove(out_filename)
            except OSError:
                pass
            raise
    finally:
        out_gguf.close()
    print(f"Converted to {out_filename}")


def write_safetensors(f, prep: PreparedConversion, opts: "ConvertOptions"):
    out_filename = _resolve_output_path(opts)
    out_st = st_mod.Safetensors(out_filename, target=True, overwrite=True)
    try:
        out_st.tensors = prep.model_tensors
        out_st.metadata = build_safetensors_metadata(f, prep.template_metadata,
                                                     prep.extra_metadata)
        sr_seed = opts.stochastic_rounding if opts.stochastic_rounding is not None else tc.DEFAULT_STOCHASTIC_SEED
        try:
            out_st.save_with_st_data(f, threads=opts.threads, groups=prep.groups,
                                     stochastic_rounding=sr_seed)
        except RuntimeError:
            try:
                os.remove(out_filename)
            except OSError:
                pass
            raise
    finally:
        out_st.close()
    print(f"Converted to {out_filename}")


# ---------------------------------------------------------------------------
# Template export / sensitivities
# ---------------------------------------------------------------------------

def filter_tensors_for_export(tensors: List[Tensor], arch_opt: Optional[Arch]) -> List[Tensor]:
    result = []
    has_model_prefix = any(t.name.startswith("model.") for t in tensors)
    for t in tensors:
        if arch_opt is not None:
            if has_model_prefix and not t.name.startswith("model."):
                continue
            if arch_opt.should_ignore(t.name):
                continue
        duped = t.dupe()
        duped.name = arch_mod.strip_prefix(duped.name)
        result.append(duped)
    return result


def write_template_from_tensors(tensors: List[Tensor], arch_opt: Optional[Arch],
                                reverse_dims: bool, writer):
    filtered = filter_tensors_for_export(tensors, arch_opt)
    tensors_obj = {}
    for t in filtered:
        if reverse_dims:
            shape = list(reversed(t.dims))
        else:
            shape = list(t.dims)
        tensors_obj[t.name] = {"shape": shape, "type": t.type}
    root = {"tensors": tensors_obj}
    json.dump(root, writer, indent=2)


def write_template_from_file(f, arch_opt: Optional[Arch], reverse_dims: bool,
                             writer):
    tensors = [t.dupe() for t in f.tensors]
    groups = tc.group_clusters(f)
    tc.collapse_model_tensors(tensors, groups, "preserve_quant")
    write_template_from_tensors(tensors, arch_opt, reverse_dims, writer)


def safetensor_type_precision(type_str: str) -> int:
    l = type_str.lower()
    if l in ("f8_e4m3", "f8_e5m2"):
        return 0
    if l in ("f16", "fp16"):
        return 1
    if l == "bf16":
        return 2
    if l in ("f32", "fp32"):
        return 3
    if l in ("f64", "fp64"):
        return 4
    return 1


def safetensor_display_type(type_str: str) -> str:
    l = type_str.lower()
    if l in ("f8_e4m3", "f8_e5m2"):
        return "FP8"
    return type_str


def template_type_suffix(template_path: str, filetype: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        parsed = json.load(f)
    t_tensors = parsed.get("tensors") or {}
    if filetype == FileType.GGUF:
        min_level = None
        min_str = "f16"
        for entry in t_tensors.values():
            type_val = entry.get("type")
            if not type_val:
                continue
            try:
                level = quantization_level_from_string(type_val)
            except ValueError:
                continue
            lv = QUANT_LEVEL_INDEX[level]
            if min_level is None or lv < min_level:
                min_level = lv
                min_str = type_val
        return min_str
    else:
        min_precision = None
        min_display = "F16"
        seen = []
        for entry in t_tensors.values():
            type_val = entry.get("type")
            if not type_val:
                continue
            display = safetensor_display_type(type_val)
            if display not in seen:
                seen.append(display)
            prec = safetensor_type_precision(type_val)
            if min_precision is None or prec < min_precision:
                min_precision = prec
                min_display = display
        if len(seen) > 1:
            return f"{min_display}-MIXED"
        return min_display


def generate_sensitivities_from_tensors(tensors: List[Tensor], arch_opt: Optional[Arch],
                                        threshold: int, writer):
    filtered = filter_tensors_for_export(tensors, arch_opt)
    sens_obj = {}
    for t in filtered:
        n_elements = 1
        for d in t.dims:
            n_elements *= d
        if n_elements < threshold:
            continue
        if arch_opt is not None:
            if arch_opt.is_high_precision(t.name):
                continue
            if arch_opt.should_upcast(t.name):
                continue
        sens_obj[t.name] = 50.0
    json.dump(sens_obj, writer, indent=2)


@dataclass
class ConvertOptions:
    path: str
    filetype: str = FileType.GGUF
    datatype: Optional[str] = None
    template_path: Optional[str] = None
    output_dir: Optional[str] = None
    output_name: Optional[str] = None
    threads: int = 1
    skip_sensitivity: bool = False
    quantization_aggressiveness: float = 50.0
    sensitivities_path: Optional[str] = None
    allowed_quant_families: Optional[QuantizationFamilies] = None
    model_only: bool = False
    allow_unknown_arch: bool = False
    allow_upscale: bool = False
    arch_override: Optional[str] = None
    stochastic_rounding: Optional[int] = None
    callbacks: Any = None
