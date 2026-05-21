# GGUFy (Python Rewrite)

This project is a Python-based fork of the original [GGUFy](https://github.com/qskousen/ggufy) project by `qskousen`. 

A lightweight, robust, and completely pure Python tool to convert model tensor formats (Safetensors to GGUF), specifically built and optimized for seamless use with [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

**Note:** This is a Python rewrite of the original Zig `ggufy` project, re-engineered for cross-platform stability, native Python GUI (Tkinter), and maximum compatibility with ComfyUI without requiring C++ compilers, Zig, or heavy PyTorch libraries.

---

## Comparison: Zig vs. Python GGUFy

| Feature / Support | Original Zig `ggufy` | Python Rewrite `GGUFy` (This Fork) |
| :--- | :--- | :--- |
| **Language & Engine** | Zig (compiled native binary) + GGML C/C++ | Pure Python 3 & NumPy |
| **Dependencies** | None (self-contained executable) | `numpy`, `gguf`, `safetensors` |
| **Cross-Platform** | Native builds for Win/Mac/Linux | Runs anywhere Python 3 + pip is available |
| **GUI Framework** | `dvui` (custom Zig GUI) | Native Tkinter (highly stable & standard) |
| **Architectures** | SD1.5, SDXL, SD3, Flux, Lumina2, Aura, HiDream, Cosmos, LTXV, Hyvid, WAN, Qwen, ERNIE | Auto-detects **SD1.5**, **SDXL**, **SD3**, and **Flux** |
| **Built-in Quantization** | Full GGML Native Quantization (`q4_k`, `q8_0`, etc.) | Seamless **`f32`**, **`f16`**, and **`bf16`** formats. *Quantization flags fallback gracefully to F16 to ensure out-of-the-box compatibility without compilation.* |
| **Sensitivity-Aware Quant** | Enabled via importance matrices | Accepts parameters (gracefully skipped in pure Python) |
| **Metadata Parsing** | Fast custom C/Zig binary parser | Pure Python zero-copy seek parser (extremely fast) |

---

## Features

- **Pure Python & NumPy:** No heavy PyTorch installations, no Zig compilers, and no C++ `safetensors` binding crashes.
- **Flawless ComfyUI Compatibility:** Automatically detects the underlying model architecture (`sd1`, `sdxl`, `sd3`, `flux`) dynamically from tensor structures and embeds the exact tags ComfyUI expects.
- **Native BF16 Support:** Implements a direct, lightning-fast byte-mapping pipeline to read `bfloat16` data seamlessly into NumPy without PyTorch.
- **Instant Metadata Loading:** Uses a custom JSON-header parser that reads structure data instantaneously without loading full model files into memory.
- **Interactive Tkinter GUI:** Includes an intuitive, lightweight GUI to convert files, inspect headers, and browse tensor structure with zero terminal commands.

## Installation

GGUFy requires Python 3.

Install the required lightweight dependencies:
```bash
pip install numpy gguf safetensors
```

*(Note: While GGUFy implements a pure python zero-copy binary parser, installing `safetensors` ensures absolute compatibility across all python runtimes).*

## GUI Usage

Run the Tkinter-based GUI using:
```bash
python ggufy_gui.py
```
The GUI allows you to easily browse for `.safetensors` files, select your target output format and datatype (e.g. `f16`, `bf16`), inspect model structure via "View Tensor Tree", and run the conversion effortlessly with real-time log outputs and progress bar tracking.

## CLI Usage

### Convert a Model
The primary use case is converting models from `.safetensors` to `.gguf` format.
```bash
python ggufy.py convert model.safetensors -d f16
```
You can optionally specify an output directory (`-o`) and output filename (`-n`):
```bash
python ggufy.py convert model.safetensors -d bf16 -n custom_name -o ./models/unet/
```

### Inspect a Model
GGUFy provides several subcommands to inspect the contents of `.safetensors` files without loading the heavy model data into memory:

**View File Header**
Displays basic tensor counts and metadata existence.
```bash
python ggufy.py header model.safetensors
```

**View Tensor Tree**
Displays a hierarchical tree of all the tensor shapes grouped by block structure. Very useful for model analysis!
```bash
python ggufy.py tree model.safetensors
```

**View Metadata**
Displays all raw file metadata key-value pairs stored in the model.
```bash
python ggufy.py metadata model.safetensors
```

### CLI Options Reference
```
-h, --help              Display help and exit
-d, --datatype          Target quantization type (default: f16). Supported: f16, f32, bf16
-f, --filetype          Target file format (default: gguf)
-o, --output-dir        Output directory (default: same as source) 
-n, --output-name       Output filename without extension
-m, --model-only        Convert only the main model (Strip 'model.' prefixes). WARNING: Leave this disabled if converting for ComfyUI.
```

## Special Thanks & Related Projects

- Original [GGUFy](https://github.com/qskousen/ggufy) project by `qskousen`.
- [ComfyUI-GGUF by city96](https://github.com/city96/ComfyUI-GGUF) for serving as the fundamental ecosystem this rewrite targets.
- [gguf library](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py) for python GGUF writing compatibility.