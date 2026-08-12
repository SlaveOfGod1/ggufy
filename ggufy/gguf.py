"""GGUF file format reader/writer.

Port of src/Gguf.zig. A Gguf object loads the header, metadata and tensor
table of a GGUF file. Writing is handled by save_with_st_data, which mirrors
the Zig writer including tracked byte counts for the size predictor.
"""

from __future__ import annotations

import io
import json
import os
import struct
from typing import Any, Dict, List, Optional

import numpy as np

from . import data_transform as dt
from . import ggml as ggml_mod
from . import types as types_mod
from .types import FileType, Tensor

ALIGNMENT = 32

GGUF_VALUE_TYPES = {
    "uint8": 0, "int8": 1, "uint16": 2, "int16": 3, "uint32": 4,
    "int32": 5, "float32": 6, "bool": 7, "string": 8, "array": 9,
    "uint64": 10, "int64": 11, "float64": 12,
}
REVERSE_VALUE_TYPES = {v: k for k, v in GGUF_VALUE_TYPES.items()}


class GgufValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class Gguf:
    formatType = FileType.GGUF

    def __init__(self, path: str, mem_allocator=None, arena_alloc=None,
                 overwrite: bool = False):
        self.path = path
        self.tensors: List[Tensor] = []
        self.metadata: Dict[str, Any] = {}
        self.version: int = 0
        self.alignment: int = ALIGNMENT
        self.data_offset: int = 0
        self.current_data_begin: int = 0
        self.file_size: int = 0
        self.file = None

        if overwrite:
            self.file = open(path, "w+b")
            self.version = 3
            self.current_data_begin = 0
            return

        if os.path.exists(path):
            self.file = open(path, "r+b")
        else:
            self.file = open(path, "w+b")
            self.version = 3
            return

        self.file_size = os.path.getsize(path)
        self.file.seek(0)
        magic = self.file.read(4)
        if magic != b"GGUF":
            self.file.close()
            raise ValueError("InvalidGgufMagic")
        self.version = struct.unpack("<I", self.file.read(4))[0]
        tensor_count = struct.unpack("<Q", self.file.read(8))[0]
        metadata_count = struct.unpack("<Q", self.file.read(8))[0]
        bytes_read = 4 + 4 + 8 + 8

        for _ in range(metadata_count):
            key, val_type = _read_metadata_header(self.file)
            bytes_read += 8 + len(key) + 4
            val, n = self._read_gguf_value_as_json(val_type)
            bytes_read += n
            self.metadata[key] = val

        for _ in range(tensor_count):
            str_len = struct.unpack("<q", self.file.read(8))[0]
            name = self.file.read(str_len).decode("utf-8", errors="replace")
            bytes_read += 8 + str_len

            dim_count = struct.unpack("<I", self.file.read(4))[0]
            bytes_read += 4
            dims = []
            for _ in range(dim_count):
                dims.append(struct.unpack("<Q", self.file.read(8))[0])
            bytes_read += 8 * dim_count

            type_id = struct.unpack("<I", self.file.read(4))[0]
            bytes_read += 4
            tensor_type = ggml_mod.GgmlType.from_int(type_id)

            tensor_offset = struct.unpack("<Q", self.file.read(8))[0]
            bytes_read += 8

            dims.reverse()  # GGUF stores innermost-first
            n_elements = 1
            for d in dims:
                n_elements *= d
            self.tensors.append(Tensor(
                name=name,
                type=tensor_type,
                dims=dims,
                size=ggml_mod.GgmlType.calc_size_in_bytes(tensor_type, n_elements),
                offset=tensor_offset,
            ))

        align_val = self.metadata.get("general.alignment", 32)
        if isinstance(align_val, bool):
            align_val = int(align_val)
        self.alignment = int(align_val)
        self.data_offset = ((bytes_read + self.alignment - 1) // self.alignment) * self.alignment
        self.current_data_begin = self.data_offset

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def open_file_for_tensor(self, name: str):
        return self.file

    def get_source_metadata(self):
        return self.metadata if self.metadata else None

    # -- reading helpers ----------------------------------------------------

    def _read_gguf_value_as_json(self, val_type: int):
        f = self.file
        if val_type == GgufValueType.BOOL:
            return f.read(1)[0] != 0, 1
        if val_type == GgufValueType.UINT8:
            return f.read(1)[0], 1
        if val_type == GgufValueType.INT8:
            return struct.unpack("<b", f.read(1))[0], 1
        if val_type == GgufValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0], 2
        if val_type == GgufValueType.INT16:
            return struct.unpack("<h", f.read(2))[0], 2
        if val_type == GgufValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0], 4
        if val_type == GgufValueType.INT32:
            return struct.unpack("<i", f.read(4))[0], 4
        if val_type == GgufValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0], 4
        if val_type == GgufValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0], 8
        if val_type == GgufValueType.INT64:
            return struct.unpack("<q", f.read(8))[0], 8
        if val_type == GgufValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0], 8
        if val_type == GgufValueType.STRING:
            str_len = struct.unpack("<q", f.read(8))[0]
            s = f.read(str_len).decode("utf-8", errors="replace")
            return s, 8 + str_len
        if val_type == GgufValueType.ARRAY:
            array_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<q", f.read(8))[0]
            n = 4 + 8
            arr = []
            for _ in range(arr_len):
                v, cn = self._read_gguf_value_as_json(array_type)
                arr.append(v)
                n += cn
            return arr, n
        raise ValueError("UnsupportedMetadataType")

    def read_gguf_version(self) -> int:
        return self.version

    def read_gguf_metadata(self) -> None:
        print(f"Metadata count: {len(self.metadata)}")
        for key, val in self.metadata.items():
            print(f"{key}: {self._print_json_value(val, 0)}")

    def read_gguf_tensor_header(self) -> None:
        print(f"Tensor count: {len(self.tensors)}")
        type_counts: Dict[str, int] = {}
        bad_size_count = 0
        for i, tensor in enumerate(self.tensors):
            ttype = tensor.type
            dims_buf = ", ".join(str(d) for d in tensor.dims)
            bad_size = False
            if ttype in ggml_mod.GGML_TYPE_IDS:
                type_counts[ttype] = type_counts.get(ttype, 0) + 1
                tensor_elements = 1
                for d in tensor.dims:
                    tensor_elements *= d
                total_bytes = ggml_mod.GgmlType.calc_size_in_bytes(ttype, tensor_elements)
                expected_padded_size = total_bytes
                if expected_padded_size % self.alignment != 0:
                    expected_padded_size += self.alignment - (expected_padded_size % self.alignment)
                actual_size = 0
                if i < len(self.tensors) - 1:
                    next_offset = self.tensors[i + 1].offset
                    allocated_size = next_offset - tensor.offset
                    actual_size = allocated_size
                    if allocated_size != expected_padded_size:
                        bad_size = True
                        bad_size_count += 1
                elif self.file_size > 0 and self.data_offset > 0:
                    start_pos = self.data_offset + tensor.offset
                    disk_size_remaining = self.file_size - start_pos
                    actual_size = disk_size_remaining
                    if disk_size_remaining != total_bytes and disk_size_remaining != expected_padded_size:
                        bad_size = True
                        bad_size_count += 1
                else:
                    print("Could not check size of last tensor: file size or data offset not set!")
                if ggml_mod.GgmlType.is_unsupported(ttype):
                    print(f"  {tensor.name}: {ttype} (Unsupported type!!!) [{dims_buf}] "
                          f"offset from tensor data start {tensor.offset}, "
                          f"offset from file start {tensor.offset + self.data_offset}")
                elif bad_size:
                    print(f"  {tensor.name}: {ttype} (BAD SIZE: actual: {actual_size}, "
                          f"expected raw: {total_bytes} expected with padding: {expected_padded_size}) "
                          f"[{dims_buf}] offset from tensor data start {tensor.offset}, "
                          f"offset from file start {tensor.offset + self.data_offset}")
                else:
                    print(f"  {tensor.name}: {ttype} [{dims_buf}] "
                          f"offset from tensor data start {tensor.offset}, "
                          f"offset from file start {tensor.offset + self.data_offset}")
            else:
                print(f"  {tensor.name}: Unknown Type [{dims_buf}] "
                      f"offset from tensor data start {tensor.offset}, "
                      f"offset from file start {tensor.offset + self.data_offset}")
        if type_counts:
            print("\n\nTensor Type Statistics:")
            for k, v in sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {k}: {v}")
        if bad_size_count > 0:
            print(f"\nTensors found with a bad size (tensor data + alignment doesn't match size on disk): {bad_size_count}\n")

    def _print_json_value(self, val, depth: int) -> str:
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, int):
            return str(val)
        if isinstance(val, float):
            return repr(val)
        if isinstance(val, str):
            return f'"{val}"'
        if isinstance(val, list):
            if len(val) > 10:
                return f"[array of {len(val)} elements]: !! Array size greater than 10, not outputting !!"
            return "[" + ", ".join(self._print_json_value(v, depth + 1) for v in val) + "]"
        if isinstance(val, dict):
            parts = []
            for k, v in val.items():
                parts.append(f'"{k}": {self._print_json_value(v, depth + 1)}')
            return "{ " + ", ".join(parts) + " }"
        return "null"

    # -- writing ------------------------------------------------------------

    def save_with_st_data(self, source, threads: int = 1, callbacks=None,
                          groups=None):
        """Write this GGUF file from an open source object (duck-typed)."""
        bytes_written = 0
        bytes_written += write_header_tracked(self.file, len(self.tensors), len(self.metadata))

        for key, val in self.metadata.items():
            bytes_written += write_metadata_kv_json_tracked(self.file, key, val)

        offset = 0
        for t in self.tensors:
            d_type = ggml_mod.GgmlType.from_string(t.type)
            bytes_written += write_tensor_info_tracked(self.file, t.name, t.dims, d_type, offset)
            next_offset = offset + t.size
            remainder = next_offset % self.alignment
            if remainder != 0:
                next_offset += (self.alignment - remainder)
            offset = next_offset

        self._maybe_write_padding(bytes_written)
        self.data_offset = self.file.tell()

        total_tensors = len(self.tensors)
        count = 1
        for t in self.tensors:
            if callbacks is not None and callbacks.is_cancelled():
                raise RuntimeError("Cancelled")
            elements = 1
            for d in t.dims:
                elements *= d
            matched = False

            # Cluster dequantization path
            if groups is not None:
                f32_buf = try_dequant_cluster(t, source, groups)
                if f32_buf is not None:
                    target_dtype = types_mod.from_string(t.type)
                    out = dt.convert_tensor_data(
                        f32_buf.astype(np.float32).tobytes(), "F32", target_dtype,
                        f32_buf.size)
                    self.file.write(out)
                    self._maybe_write_padding(calculate_tensor_size(t))
                    if callbacks is not None:
                        callbacks.report_progress(count, total_tensors, t.name, "nvfp4", t.type, elements)
                    count += 1
                    matched = True

            if not matched:
                for source_tensor in source.tensors:
                    if source_tensor.name == t.name or (
                        len(source_tensor.name) > len(t.name)
                        and source_tensor.name[len(source_tensor.name) - len(t.name) - 1] == '.'
                        and source_tensor.name.endswith(t.name)
                    ):
                        matched = True
                        source_dtype = types_mod.from_string(source_tensor.type)
                        n_elements = 1
                        for d in t.dims:
                            n_elements *= d
                        source_size = types_mod.calc_size_in_bytes(source_dtype, n_elements)
                        src = _read_source_tensor(source, source_tensor, source_size)
                        self.write_tensor_data(t, source_dtype, src)
                        self._maybe_write_padding(calculate_tensor_size(t))
                        if callbacks is not None:
                            callbacks.report_progress(count, total_tensors, t.name, source_tensor.type, t.type, elements)
                        count += 1
                        break

            if not matched:
                raise RuntimeError(f"NoMatchingSourceTensor: {t.name}")

        self.file.flush()

    def _maybe_write_padding(self, size: int):
        padding_len = (self.alignment - (size % self.alignment)) % self.alignment
        if padding_len:
            self.file.write(b"\x00" * padding_len)

    def write_tensor_data(self, t: Tensor, source_dtype: str, source_data: bytes):
        target_dtype = types_mod.from_string(t.type)
        n_elements = 1
        for d in t.dims:
            n_elements *= d
        if types_mod.equivalent_type(source_dtype, target_dtype):
            self.file.write(source_data)
        else:
            converted = dt.convert_tensor_data(source_data, source_dtype, target_dtype, n_elements)
            self.file.write(converted)

    def write_template(self, writer):
        root = {"metadata": self.metadata, "tensors": {}}
        for t in self.tensors:
            shape = list(reversed(t.dims))
            root["tensors"][t.name] = {"shape": shape, "type": t.type}
        json.dump(root, writer, indent=2)


# ---------------------------------------------------------------------------
# Module-level helpers (mirroring Gguf module functions)
# ---------------------------------------------------------------------------

def _read_metadata_header(f):
    title_len = struct.unpack("<Q", f.read(8))[0]
    if title_len == 0 or title_len > 1024 * 1024:
        raise ValueError("InvalidMetadataTitleLength")
    title = f.read(title_len).decode("utf-8", errors="replace")
    val_type = struct.unpack("<I", f.read(4))[0]
    return title, val_type


def write_header_tracked(writer, tensor_count: int, metadata_count: int) -> int:
    writer.write(b"GGUF")
    writer.write(struct.pack("<I", 3))
    writer.write(struct.pack("<Q", tensor_count))
    writer.write(struct.pack("<Q", metadata_count))
    return 4 + 4 + 8 + 8


def write_string_tracked(writer, s: str) -> int:
    data = s.encode("utf-8")
    writer.write(struct.pack("<Q", len(data)))
    writer.write(data)
    return 8 + len(data)


def write_metadata_kv_json_tracked(writer, key: str, value) -> int:
    bytes_written = write_string_tracked(writer, key)
    if isinstance(value, bool):
        writer.write(struct.pack("<I", GgufValueType.BOOL))
        bytes_written += 4
        writer.write(b"\x01" if value else b"\x00")
        bytes_written += 1
    elif isinstance(value, int):
        if 0 <= value <= 0xFFFFFFFF:
            writer.write(struct.pack("<I", GgufValueType.UINT32))
            bytes_written += 4
            writer.write(struct.pack("<I", value))
            bytes_written += 4
        else:
            writer.write(struct.pack("<I", GgufValueType.INT64))
            bytes_written += 4
            writer.write(struct.pack("<q", value))
            bytes_written += 8
    elif isinstance(value, float):
        writer.write(struct.pack("<I", GgufValueType.FLOAT32))
        bytes_written += 4
        writer.write(struct.pack("<f", value))
        bytes_written += 4
    elif isinstance(value, str):
        writer.write(struct.pack("<I", GgufValueType.STRING))
        bytes_written += 4
        bytes_written += write_string_tracked(writer, value)
    elif isinstance(value, list):
        if len(value) == 0:
            raise ValueError("EmptyMetadataArray")
        first = value[0]
        if isinstance(first, bool):
            array_type = GgufValueType.BOOL
        elif isinstance(first, int):
            array_type = GgufValueType.INT32
        elif isinstance(first, float):
            array_type = GgufValueType.FLOAT32
        elif isinstance(first, str):
            array_type = GgufValueType.STRING
        else:
            raise ValueError("UnsupportedMetadataArrayType")
        writer.write(struct.pack("<I", GgufValueType.ARRAY))
        bytes_written += 4
        writer.write(struct.pack("<I", array_type))
        bytes_written += 4
        writer.write(struct.pack("<Q", len(value)))
        bytes_written += 8
        for item in value:
            if array_type == GgufValueType.BOOL:
                writer.write(b"\x01" if item else b"\x00")
                bytes_written += 1
            elif array_type == GgufValueType.INT32:
                writer.write(struct.pack("<i", int(item)))
                bytes_written += 4
            elif array_type == GgufValueType.FLOAT32:
                writer.write(struct.pack("<f", float(item)))
                bytes_written += 4
            elif array_type == GgufValueType.STRING:
                bytes_written += write_string_tracked(writer, str(item))
    else:
        raise ValueError("UnsupportedMetadataType")
    return bytes_written


def write_tensor_info_tracked(writer, name: str, dims, type_: str, offset: int) -> int:
    bytes_written = write_string_tracked(writer, name)
    writer.write(struct.pack("<I", len(dims)))
    bytes_written += 4
    for d in reversed(dims):
        writer.write(struct.pack("<Q", d))
        bytes_written += 8
    writer.write(struct.pack("<I", ggml_mod.GGML_TYPE_IDS[type_]))
    bytes_written += 4
    writer.write(struct.pack("<Q", offset))
    bytes_written += 8
    return bytes_written


def calculate_tensor_size(t: Tensor) -> int:
    ggml_type = ggml_mod.GgmlType.from_string(t.type)
    n_elements = 1
    for d in t.dims:
        n_elements *= d
    return ggml_mod.GgmlType.calc_size_in_bytes(ggml_type, n_elements)


def calculate_file_size(tensors: List[Tensor], metadata: Dict[str, Any],
                        alignment: int = 32) -> int:
    buf = io.BytesIO()
    total = write_header_tracked(buf, len(tensors), len(metadata))
    for key, val in metadata.items():
        total += write_metadata_kv_json_tracked(buf, key, val)
    offset = 0
    for t in tensors:
        d_type = ggml_mod.GgmlType.from_string(t.type)
        total += write_tensor_info_tracked(buf, t.name, t.dims, d_type, offset)
        next_offset = offset + t.size
        remainder = next_offset % alignment
        if remainder != 0:
            next_offset += (alignment - remainder)
        offset = next_offset
    total += (alignment - (total % alignment)) % alignment
    for t in tensors:
        size = calculate_tensor_size(t)
        total += size + (alignment - (size % alignment)) % alignment
    return total


def _read_source_tensor(source, tensor: Tensor, size: int) -> bytes:
    f = source.open_file_for_tensor(tensor.name)
    pos = tensor.offset + source.current_data_begin
    f.seek(pos)
    return f.read(size)


def try_dequant_cluster(t: Tensor, source, groups):
    """Return F32 numpy array for a cluster-sourced dest tensor, or None."""
    if groups is None:
        return None
    from . import tensor_clusters
    return tensor_clusters.try_dequant_cluster(t, source, groups)
