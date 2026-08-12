"""SafeTensors file format reader/writer.

Port of src/Safetensor.zig. Supports single files and sharded
(model.safetensors.index.json / directory) layouts. The writer emits the
canonical uppercase dtype spellings and coerces metadata values to strings,
matching the reference Rust implementation's expectations.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional

import numpy as np

from . import data_transform as dt
from . import tensor_clusters as tc
from . import types as types_mod
from .types import FileType, Tensor

ST_DTYPE_SIZES = {
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U32": 4, "F64": 8,
    "I64": 8, "U64": 8, "I8": 1, "U8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "SCALED_F8_E4M3": 1, "I16": 2, "U16": 2, "F4_E2M1": 1, "MXFP4": 1,
    "MXFP8_E4M3": 1, "NVFP4": 1,
}


class DType:
    NAMES = list(ST_DTYPE_SIZES.keys())

    @staticmethod
    def from_string(s: str) -> str:
        upper = s.upper()
        if upper not in ST_DTYPE_SIZES:
            raise ValueError(f"UnknownDType: {s}")
        return upper

    @staticmethod
    def get_size_in_bytes(t: str) -> int:
        return ST_DTYPE_SIZES[t]

    @staticmethod
    def calc_size_in_bytes(t: str, n: int) -> int:
        if t in ("F4_E2M1", "MXFP4", "NVFP4"):
            return (n + 1) // 2
        return ST_DTYPE_SIZES[t] * n


def safetensors_dtype_name(type_str: str) -> str:
    dt = types_mod.from_string(type_str)
    stype = types_mod.for_format(dt, FileType.SAFETENSORS)
    return stype


def stringify_metadata_values(meta: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for key, value in meta.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif value is None:
            out[key] = "null"
        elif isinstance(value, int):
            out[key] = str(value)
        elif isinstance(value, float):
            out[key] = repr(value)
        else:
            out[key] = json.dumps(value, separators=(",", ":"))
    return out


def build_header_object(metadata: Optional[Dict[str, Any]],
                        tensors: List[Tensor]) -> Dict[str, Any]:
    header_obj: Dict[str, Any] = {}
    if metadata:
        header_obj["__metadata__"] = stringify_metadata_values(metadata)

    for t in tensors:
        dt_type = types_mod.from_string(t.type)
        specs = tc.cluster_write_layout(dt_type, t.dims)
        if specs is not None:
            base = t.name[:-len(".weight")] if t.name.endswith(".weight") else t.name
            off = t.offset
            for s in specs:
                name = base + s.suffix
                header_obj[name] = {
                    "dtype": s.dtype,
                    "shape": list(s.dims),
                    "data_offsets": [off, off + s.bytes],
                }
                off += s.bytes
            continue

        header_obj[t.name] = {
            "dtype": safetensors_dtype_name(t.type),
            "shape": list(t.dims),
            "data_offsets": [t.offset, t.offset + t.size],
        }
    return header_obj


def report_unwritable_dtype(tensors: List[Tensor]):
    for t in tensors:
        try:
            safetensors_dtype_name(t.type)
        except ValueError:
            print(f"Tensor {t.name} has type {t.type}, which has no SafeTensors "
                  f"representation. Block-quantized types (q4_k, q6_k, q8_0, ...) "
                  f"can only be written to a GGUF file — use -f gguf.")
            return


class Safetensors:
    def __init__(self, path: str, mem_allocator=None, arena_alloc=None,
                 target: bool = False, overwrite: bool = False):
        self.path = path
        self.tensors: List[Tensor] = []
        self.metadata: Optional[Dict[str, Any]] = None
        self.current_file_handle = None
        self.current_open_path = ""
        self.current_data_begin = 0

        if target:
            self._open_target(path, overwrite)
            return

        entry_path = path
        if os.path.isdir(path):
            index_path = os.path.join(path, "model.safetensors.index.json")
            single_path = os.path.join(path, "model.safetensors")
            if os.path.exists(index_path):
                entry_path = index_path
            elif os.path.exists(single_path):
                entry_path = single_path
            else:
                raise FileNotFoundError("ModelNotFound")

        if entry_path.endswith("index.json"):
            self._load_sharded(entry_path)
        else:
            self._load_single(entry_path)

    def _open_target(self, path: str, overwrite: bool):
        mode = "w+b" if overwrite else "xb"
        try:
            self.current_file_handle = open(path, mode)
        except FileExistsError:
            raise FileExistsError(f"File already exists: {path}")
        self.current_open_path = path

    def close(self):
        if self.current_file_handle:
            self.current_file_handle.close()
            self.current_file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _load_single(self, path: str):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            self.current_data_begin = header_len + 8
            header_bytes = f.read(header_len)
        root = json.loads(header_bytes.decode("utf-8"))
        if "__metadata__" in root:
            self.metadata = root["__metadata__"]
        self._extract_tensors_from_object(root, path)

    def _load_sharded(self, index_path: str):
        directory = os.path.dirname(index_path) or "."
        with open(index_path, "rb") as f:
            index_json = json.loads(f.read().decode("utf-8"))
        filenames = []
        weight_map = index_json.get("weight_map", {})
        for fname in weight_map.values():
            if fname not in filenames:
                filenames.append(fname)
        first = True
        for fname in filenames:
            full_path = os.path.join(directory, fname)
            with open(full_path, "rb") as shard_file:
                header_len = struct.unpack("<Q", shard_file.read(8))[0]
                self.current_data_begin = header_len + 8
                header_bytes = shard_file.read(header_len)
            shard_json = json.loads(header_bytes.decode("utf-8"))
            if first:
                if "__metadata__" in shard_json:
                    self.metadata = shard_json["__metadata__"]
                first = False
            self._extract_tensors_from_object(shard_json, full_path)

    def _extract_tensors_from_object(self, root: Dict[str, Any], source_path: str):
        for name, obj in root.items():
            if name == "__metadata__":
                continue
            dtype = DType.from_string(obj["dtype"])
            shape = [int(d) for d in obj["shape"]]
            offsets = obj["data_offsets"]
            start = int(offsets[0])
            end = int(offsets[1])
            self.tensors.append(Tensor(
                name=name,
                type=dtype,
                dims=shape,
                size=end - start,
                offset=start,
                source_path=source_path,
            ))

    def open_file_for_tensor(self, name: str):
        for t in self.tensors:
            if t.name == name:
                tensor_path = t.source_path or self.path
                if self.current_open_path != tensor_path:
                    if self.current_file_handle:
                        self.current_file_handle.close()
                    self.current_file_handle = open(tensor_path, "rb")
                    self.current_open_path = tensor_path
                    self.current_file_handle.seek(0)
                    st_len = struct.unpack("<Q", self.current_file_handle.read(8))[0]
                    self.current_data_begin = 8 + st_len
                return self.current_file_handle
        raise KeyError(f"TensorNotFound: {name}")

    def get_source_metadata(self):
        return self.metadata

    # -- display commands ----------------------------------------------------

    def print_metadata(self):
        if self.metadata:
            for key, value in self.metadata.items():
                if isinstance(value, str) and value and value[0] in "{[":
                    try:
                        nested = json.loads(value)
                        print(f"{key}: {json.dumps(nested, indent=2)}")
                    except json.JSONDecodeError:
                        print(f"{key}: {value}")
                elif isinstance(value, str):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {json.dumps(value, indent=2)}")
        else:
            print("No metadata found.")

    def print_header(self):
        dtype_counts: Dict[str, int] = {}
        bad_offset_count = 0
        print("{")
        for i, t in enumerate(self.tensors):
            dt = DType.from_string(t.type)
            bad_offset = False
            n_elements = 1
            for d in t.dims:
                n_elements *= d
            expected_size = DType.calc_size_in_bytes(dt, n_elements)
            if t.size != expected_size:
                print(f"  WARNING Tensor {t.name}: stored size {t.size} does not "
                      f"match expected size {expected_size} (dtype={t.type}, elements={n_elements})")
                bad_offset = True
                bad_offset_count += 1
            if i < len(self.tensors) - 1:
                next_offset = self.tensors[i + 1].offset
                allocated_size = next_offset - t.offset
                if allocated_size != expected_size:
                    print(f"  WARNING Tensor {t.name}: allocated region {allocated_size} "
                          f"does not match expected size {expected_size}")
                    bad_offset = True
                    bad_offset_count += 1
            else:
                if self.current_data_begin > 0:
                    pass
                else:
                    print(f"  WARNING Tensor {t.name}: cannot validate last tensor offset, data_begin not set")

            flag = " <-- BAD OFFSET" if bad_offset else ""
            dims_str = ", ".join(str(d) for d in t.dims)
            print(f'  "{t.name}": {{')
            print(f'    "dtype": "{t.type}",')
            print(f'    "shape": [{dims_str}],')
            print(f'    "offset_from_data_start_and_file_start": [{t.offset}, {t.offset + self.current_data_begin}]{flag}')
            print("  }" if i == len(self.tensors) - 1 else "  },")
            dtype_counts[t.type] = dtype_counts.get(t.type, 0) + 1
        print("}")
        if dtype_counts:
            print("\n\nTensor Type Statistics:")
            for k, v in dtype_counts.items():
                print(f"  {k}: {v}")
        if bad_offset_count > 0:
            print(f"\nTensors with bad offsets/sizes: {bad_offset_count}")

    def print_tensor_tree(self):
        root = _TreeNode("root")
        for t in self.tensors:
            parts = t.name.split(".")
            current = root
            for part in parts:
                current = current.children.setdefault(part, _TreeNode(part, current))
            current.dtype = DType.from_string(t.type)
            current.shape = list(t.dims)
            current.data_offsets = [t.offset, t.offset + t.size]
        self._print_node(root, 0)

    def _print_node(self, node, depth):
        print("  " * depth + node.name, end="")
        if len(node.children) == 0:
            if node.dtype is not None:
                parts = [f"dtype: {node.dtype}"]
                if node.shape is not None:
                    parts.append(f"shape: [{', '.join(str(d) for d in node.shape)}]")
                if node.data_offsets is not None:
                    parts.append(f"offsets: [{node.data_offsets[0]}, {node.data_offsets[1]}]")
                if node.shape is not None:
                    total = 1
                    for d in node.shape:
                        total *= d
                    parts.append(f"size: {DType.calc_size_in_bytes(node.dtype, total)}")
                print(" (" + ", ".join(parts) + ")", end="")
        print()
        for key in sorted(node.children.keys()):
            self._print_node(node.children[key], depth + 1)

    # -- writing ------------------------------------------------------------

    def save_with_st_data(self, source, threads: int = 1, callbacks=None,
                          groups=None, stochastic_rounding: int = 0xC0FFEE):
        header_obj = build_header_object(self.metadata, self.tensors)
        header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
        header_size = len(header_bytes)

        w = self.current_file_handle
        w.seek(0)
        w.truncate()
        w.write(struct.pack("<Q", header_size))
        w.write(header_bytes)

        total_tensors = len(self.tensors)
        count = 1
        for t in self.tensors:
            if callbacks is not None and callbacks.is_cancelled():
                raise RuntimeError("Cancelled")
            elements = 1
            for d in t.dims:
                elements *= d
            matched = False

            dt_type = types_mod.from_string(t.type)
            dest_is_cluster = tc.is_cluster_type(dt_type)
            src_f32 = None
            if dest_is_cluster:
                src_f32 = tc.dequant_source_to_f32(t, source, groups)
            else:
                src_f32 = tc.try_dequant_cluster(t, source, groups)

            if src_f32 is not None:
                if dest_is_cluster:
                    tc.write_cluster_data(w, dt_type, src_f32, t.dims, stochastic_rounding)
                else:
                    out = dt.convert_tensor_data(
                        np.ascontiguousarray(src_f32, dtype=np.float32).tobytes(),
                        "F32", dt_type, src_f32.size)
                    w.write(out)
                if callbacks is not None:
                    callbacks.report_progress(count, total_tensors, t.name, "cluster", t.type, elements)
                count += 1
                matched = True

            if not matched:
                best_idx = tc.best_source_match(source.tensors, t.name)
                if best_idx is not None:
                    source_tensor = source.tensors[best_idx]
                    matched = True
                    source_dtype = types_mod.from_string(source_tensor.type)
                    n_elements = 1
                    for d in t.dims:
                        n_elements *= d
                    source_size = types_mod.calc_size_in_bytes(source_dtype, n_elements)
                    sf = source.open_file_for_tensor(source_tensor.name)
                    sf.seek(source_tensor.offset + source.current_data_begin)
                    src_bytes = sf.read(source_size)
                    self.write_tensor_data(t, source_dtype, src_bytes)
                    if callbacks is not None:
                        callbacks.report_progress(count, total_tensors, t.name,
                                                  source_tensor.type, t.type, elements)
                    count += 1

            if not matched:
                raise RuntimeError(f"NoMatchingSourceTensor: {t.name}")

        w.flush()

    def write_tensor_data(self, t: Tensor, source_dtype: str, source_data: bytes):
        target_dtype = types_mod.from_string(t.type)
        n_elements = 1
        for d in t.dims:
            n_elements *= d
        if types_mod.equivalent_type(source_dtype, target_dtype):
            self.current_file_handle.write(source_data)
        else:
            converted = dt.convert_tensor_data(source_data, source_dtype, target_dtype, n_elements)
            self.current_file_handle.write(converted)


class _TreeNode:
    def __init__(self, name: str, parent=None):
        self.name = name
        self.children: Dict[str, _TreeNode] = {}
        self.parent = parent
        self.dtype = None
        self.shape = None
        self.data_offsets = None


def calculate_file_size(tensors: List[Tensor], metadata: Optional[Dict[str, Any]],
                        arena_alloc=None, allocator=None) -> int:
    try:
        header_obj = build_header_object(metadata, tensors)
    except ValueError:
        report_unwritable_dtype(tensors)
        raise
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
    header_size = len(header_bytes)
    data_size = sum(t.size for t in tensors)
    return 8 + header_size + data_size
