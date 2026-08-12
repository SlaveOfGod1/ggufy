"""Unified model-file loader used by the GUI and CLI inspection commands.

Port of src/FileLoader.zig: detects the file format, loads tensors + metadata,
builds a per-type count summary and detects the architecture.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import image_arch as arch_mod
from .gguf import Gguf
from .safetensor import Safetensors
from .types import FileType


class TensorFile:
    def __init__(self):
        self.type = FileType.SAFETENSORS
        self.arch = None
        self.tensors = []
        self.metadata = None
        self.size_in_bytes = 0
        self.type_counts: Dict[str, int] = {}
        self.types_line = ""
        # The underlying open reader (Safetensors or Gguf). Kept alive so the
        # conversion pipeline can read tensor data through the duck-typed
        # source interface (open_file_for_tensor / current_data_begin /
        # get_source_metadata).
        self._source = None

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None

    def get_source_metadata(self):
        if self._source is not None:
            return self._source.get_source_metadata()
        return self.metadata

    def open_file_for_tensor(self, name: str):
        if self._source is None:
            raise ValueError("No source file open")
        return self._source.open_file_for_tensor(name)

    @property
    def current_data_begin(self) -> int:
        if self._source is not None:
            return self._source.current_data_begin
        return 0

    @staticmethod
    def load_file(path: str) -> "TensorFile":
        ret = TensorFile()
        ret.size_in_bytes = os.path.getsize(path)

        with open(path, "rb") as f:
            try:
                ret.type = FileType.detect_from_file(f)
            except ValueError:
                ret.type = FileType.SAFETENSORS

        if ret.type == FileType.SAFETENSORS:
            sf = Safetensors(path)
            ret._source = sf
            ret.metadata = sf.metadata
            ret.tensors = sf.tensors
        else:
            g = Gguf(path)
            ret._source = g
            ret.metadata = g.metadata
            ret.tensors = g.tensors

        for tensor in ret.tensors:
            ret.type_counts[tensor.type] = ret.type_counts.get(tensor.type, 0) + 1

        if ret.type_counts:
            parts = []
            for k, v in ret.type_counts.items():
                parts.append(f"{k} {v}")
            ret.types_line = " ".join(parts)

        ret.arch = arch_mod.detect_arch_from_tensors(ret.tensors)
        return ret
