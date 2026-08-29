"""ggufy (Anima edition) - a lightweight and efficient tool to convert the
official Anima model (circlestone-labs/Anima) between SafeTensors and GGUF.

Python port of the Zig ggufy tool, specialized for Anima in this fork: see
`ggufy/image_arch.py` and `README.md` for the Anima-only scope and rationale.
"""

__version__ = "0.1.0"

from . import convert, data_transform, ggml, image_arch, safetensor, tensor_clusters, types
from .file_loader import TensorFile
