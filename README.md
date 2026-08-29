# ggufy
A lightweight and efficient tool to convert tensor formats.

ggufy:
- is a pure-Python port (Python 3 + NumPy) of the original [ggufy](https://github.com/qskousen/ggufy), which works on linux, windows, and macos
- comes in CLI and GUI flavors
- supports converting from safetensors to various gguf quantizations
- supports converting safetensors datatypes (F32, BF16, F16, F8 E4M3/E5M2, Scaled F8 E4M3, MXFP8 E4M3, NVFP4, INT8, INT8 CONVROT, INT4 CONVROT, INT4 CONVROT SR, ASYM_W4A8_INT8, MXFP4)
- supports converting with "[quantization sensitivity](docs/CLI.md#sensitivity-aware-quantization)" files (some architectures built-in)
- currently targets image and video diffusion models (SD1.5, SDXL, Anima, etc.)

This fork adds support for the official [Anima](https://huggingface.co/circlestone-labs/Anima) model (a 2B text-to-image model built on NVIDIA Cosmos-Predict2 with a bolted-on T5 text adapter).

### Supported architectures

This table lists the architectures that ggufy can convert, and whether they have sensitivity data available.

| Architecture       | Supported | Sensitivity Data |
|--------------------|-----------|------------------|
| SD1.5              | ✅         | ✅                |
| SDXL               | ✅         | ✅                |
| Flux               | ✅         | ❌                |
| Lumina2 (ZiT, ZiB) | ✅         | ❌                |
| Aura               | ✅         | ❌                |
| HiDream            | ✅         | ❌                |
| Cosmos             | ✅         | ❌                |
| Anima              | ✅         | ❌                |
| LTXV               | ✅         | ❌                |
| LTX2               | ✅         | ❌                |
| Hyvid              | ✅         | ❌                |
| WAN                | ✅         | ❌                |
| SD3                | ✅         | ❌                |
| Qwen               | ✅         | ❌                |
| ERNIE              | ✅         | ❌                |
| Krea2              | ✅         | ❌                |
| Mage-Flow          | ✅         | ❌                |

## Installation

This is a Python port, so the only requirement is Python 3.8+ and `numpy`.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install numpy
```

Run directly from the repo:

```bash
python main.py --help
```

Or install as a console script:

```bash
pip install .
ggufy --help
```

## Usage

The primary use case for ggufy is converting models from safetensors to gguf format:

```bash
ggufy convert model.safetensors -d q4_k
```

There are three main ways to convert:
- using the `convert` command by itself,
- using the `template` command to generate a JSON template for a model and using it with `convert`,
- using the `convert` command with a sensitivity file to perform sensitivity-aware quantization.

### Converting Anima

Anima is detected automatically from its tensor keys (`blocks.0.mlp.layer1.weight`,
`blocks.0.adaln_modulation_cross_attn.1.weight`, and the discriminator
`llm_adapter.blocks.0.cross_attn.q_proj.weight`). The entire `llm_adapter`
(embedding + cross/self-attention blocks) is kept high-precision, since its
`embed.weight` is an `nn.Embedding` that cannot be block-quantized.

```bash
# Convert Anima to GGUF
python main.py convert anima-base-v1.0.safetensors -d q4_k

# With a specific output name and directory
python main.py convert anima-base-v1.0.safetensors -d q8_0 -n anima-base-q8_0 -o ./converted/
```

### CLI

For the full command reference — all commands and options, quantization levels, sensitivity-aware quantization, inspecting model files, and complete examples — see **[docs/CLI.md](docs/CLI.md)**.

```bash
ggufy convert model.safetensors -d q4_k -n my-model-q4-k -o ./converted/
ggufy convert sdxl.safetensors -d q4_k -a 25          # sensitivity, conservative
ggufy convert sdxl.safetensors -d q4_k -x             # skip sensitivity
ggufy template existing.gguf                          # export template.json
ggufy convert model.safetensors -t template.json      # convert using template
ggufy header model.safetensors
ggufy tree model.safetensors
ggufy metadata model.gguf
ggufy names model.safetensors
ggufy sensitivities model.safetensors
ggufy version
```

### Options

```
-h, --help              Display help and exit
-d, --datatype          Target quantization type (default: source datatype)
-f, --filetype          Target file format (default: gguf) Options: gguf, safetensors
-t, --template          Use a JSON template for conversion
-o, --output-dir        Output directory (default: same as source)
-n, --output-name       Output filename without extension (default: source name + datatype)
-j, --threads           Number of threads for quantization (default: CPU core count)
-a, --aggressiveness    Aggressiveness of sensitivity-aware quantization (default: 50)
-x, --skip-sensitivity  Skip sensitivity-aware quantization
-s, --sensitivities     Path to a sensitivities JSON file to use
-q, --use-quant-types   Quantization families to use with sensitivity (e.g. "k", "0,k")
-m, --model-only        When output is safetensors, convert only the main model
-u, --allow-unknown-arch Allow converting files with unrecognized architectures
-U, --allow-upscale     Allow converting from a lower-precision source to a higher-precision target
-A, --arch              Set the architecture name written to GGUF metadata
-R, --stochastic-rounding Seed for INT4_CONVROT_SR stochastic rounding
-c, --calculate-size    Compute and print the exact final output size without writing
-S, --shapes            With names: emit {"name":...,"shape":[...]} objects
```

## GUI

```bash
python main.py gui
# or after pip install: ggufy-gui
```

A tkinter GUI (no extra dependencies) mirroring the original app.

## Acknowledgements

- [ggufy (Zig)](https://github.com/qskousen/ggufy) — the original tool this port is based on
- [ggml](https://github.com/ggml-org/ggml) — quantization algorithms ported from the reference implementation
- [ComfyUI-GGUF by city96](https://github.com/city96/ComfyUI-GGUF) — for helping me to understand a lot about how the quantization works, as well as architecture detection
- [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant) — reference for Anima quantization rules
