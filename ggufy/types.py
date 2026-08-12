"""Data type definitions shared across the ggufy package.

Mirrors src/types.zig: the DataType vocabulary (safetensors + ggml), the
FileType enum, and the Tensor struct used throughout the pipeline.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional


class FileType:
    SAFETENSORS = "safetensors"
    GGUF = "gguf"

    @staticmethod
    def detect_from_file(fileobj) -> str:
        """Read 8 bytes from an open binary file at the start and detect the format.

        Consumes the first 8 bytes of the stream.
        """
        header = fileobj.read(8)
        if len(header) < 8:
            raise ValueError("Unknown format")
        if header[0:4] == b"GGUF":
            return FileType.GGUF
        (possible_length,) = struct.unpack("<Q", header[0:8])
        if 0 < possible_length < 128 * 1024 * 1024:  # 128 MiB cap
            return FileType.SAFETENSORS
        raise ValueError("Unknown format")

    @staticmethod
    def parse_from_string(s: str) -> str:
        if s == "safetensors":
            return FileType.SAFETENSORS
        if s == "gguf":
            return FileType.GGUF
        raise ValueError("Unknown format")


@dataclass
class Tensor:
    name: str
    type: str
    dims: List[int]
    size: int
    offset: int
    source_path: Optional[str] = None

    def dupe(self) -> "Tensor":
        return Tensor(
            name=self.name,
            type=self.type,
            dims=list(self.dims),
            size=self.size,
            offset=self.offset,
            source_path=self.source_path,
        )


# Safetensors-native data types (uppercase spelling)
ST_TYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "SCALED_F8_E4M3",
    "F4_E2M1",
    "MXFP4",
    "MXFP8_E4M3",
    "NVFP4",
    "INT8",
    "INT8_CONVROT",
    "INT4_CONVROT",
    "INT4_CONVROT_SR",
    "ASYM_W4A8_INT8",
    "BF16",
    "F16",
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
}

# GGML types (lowercase spelling)
GGML_TYPES = {
    "f32", "f16", "q4_0", "q4_1", "q4_2", "q4_3", "q5_0", "q5_1", "q8_0",
    "q8_1", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "q8_k", "iq2_xxs",
    "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs",
    "i8", "i16", "i32", "i64", "f64", "iq1_m", "bf16", "q4_0_4_4",
    "q4_0_4_8", "q4_0_8_8", "tq1_0", "tq2_0", "iq4_nl_4_4", "iq4_nl_4_8",
    "iq4_nl_8_8", "mxfp4", "nvfp4", "q1_0",
}

ALL_TYPES = ST_TYPES | GGML_TYPES | {"count"}

# Equivalent (safetensors, gguf) type pairs.
EQUIVALENCE_TABLE = [
    ("F16", "f16"),
    ("F32", "f32"),
    ("F64", "f64"),
    ("BF16", "bf16"),
    ("I8", "i8"),
    ("I16", "i16"),
    ("I32", "i32"),
    ("I64", "i64"),
]


def format_type(dt: str) -> str:
    if dt in GGML_TYPES:
        return FileType.GGUF
    return FileType.SAFETENSORS


def from_string(s: str) -> str:
    if s not in ALL_TYPES:
        raise ValueError(f"InvalidDataType: {s}")
    return s


def for_format(dt: str, filetype: str) -> str:
    """Convert dt to the equivalent type name for the given file format."""
    if format_type(dt) == filetype:
        return dt
    for a, b in EQUIVALENCE_TABLE:
        if dt == a:
            return b
        if dt == b:
            return a
    raise ValueError(f"NoEquivalentType: {dt}")


def equivalent_type(dt: str, target: str) -> bool:
    """True if dt and target represent the same underlying data type."""
    if target not in ALL_TYPES:
        return False
    if format_type(dt) == format_type(target):
        return dt == target
    st_type = dt if format_type(dt) == FileType.SAFETENSORS else target
    gg_type = dt if format_type(dt) == FileType.GGUF else target
    return (st_type, gg_type) in EQUIVALENCE_TABLE


def calc_size_in_bytes(dt: str, n_elements: int) -> int:
    """Byte size for `n_elements` of the given DataType."""
    if dt in ("SCALED_F8_E4M3", "INT8_CONVROT", "INT8"):
        return n_elements
    if dt in ("INT4_CONVROT", "INT4_CONVROT_SR", "ASYM_W4A8_INT8"):
        return (n_elements + 1) // 2
    if format_type(dt) == FileType.SAFETENSORS:
        from .safetensor import DType
        return DType.calc_size_in_bytes(dt, n_elements)
    from .ggml import GgmlType
    return GgmlType.calc_size_in_bytes(dt, n_elements)


def precision_rank(type_str: str) -> int:
    """Rough precision rank - higher means more information per element.
    255 for types that are not meaningful to compare."""
    if type_str not in ALL_TYPES:
        return 255
    if type_str in ("q1_0", "tq1_0", "iq1_s", "iq1_m"):
        return 1
    if type_str in ("q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "tq2_0"):
        return 2
    if type_str in ("q3_k", "iq3_xxs", "iq3_s"):
        return 3
    if type_str in ("q4_0", "q4_1", "q4_k", "iq4_nl", "iq4_xs",
                    "nvfp4", "mxfp4", "NVFP4", "MXFP4", "F4_E2M1"):
        return 4
    if type_str in ("q5_0", "q5_1", "q5_k"):
        return 5
    if type_str == "q6_k":
        return 6
    if type_str in ("q8_0", "q8_1", "q8_k", "F8_E4M3", "F8_E5M2",
                    "SCALED_F8_E4M3", "MXFP8_E4M3"):
        return 7
    if type_str in ("f16", "F16", "bf16", "BF16", "i8", "I8", "I16",
                    "i16", "U8", "U16"):
        return 8
    if type_str in ("f32", "F32", "i32", "I32", "U32"):
        return 9
    if type_str in ("f64", "F64", "i64", "I64", "U64"):
        return 10
    return 255
