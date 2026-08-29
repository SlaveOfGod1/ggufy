# ggufy (Python)

A Python port of [ggufy](https://github.com/qskousen/ggufy) — a lightweight and
efficient tool to convert tensor formats (SafeTensors <-> GGUF) with full
GGML quantization, sensitivity-aware quantization, ComfyUI cluster support,
and a tkinter GUI.

This branch adds **Anima support** for the official
[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) model.

## Supported architectures

| Architecture | Model family | Notes |
|---|---|---|
| **flux** | FLUX.1, Black Forest Labs | Schnell, Dev, Pro |
| **sd3** | Stable Diffusion 3 | Multi-rectified flow |
| **aura** | AuraFlow | Visual flow |
| **hidream** | HiDream | Text-to-image |
| **ltxv** | LTX-Video | Video generation |
| **hyvid** | Hybrid Video | Video diffusion |
| **wan** | Wan | Video generation |
| **sdxl** | Stable Diffusion XL | SDXL, SDXL Turbo, Pony, Illustrious, NovaeAnime etc. |
| **sd1** | Stable Diffusion 1.5 | SD1.5, RealisticVision, CyberRealistic etc. |
| **lumina2** | Lumina Image 2.0 | Gemma2-based |
| **mage_flow** | MAGE | Flow-based |
| **qwen** | Qwen | Multimodal |
| **ernie** | ERNIE | Baidu |
| **krea2** | Krea 2 | Visual flow |
| **cosmos** | NVIDIA Cosmos-Predict2 | MiniTrainDIT backbone |
| **anima** | circlestone-labs/Anima | Cosmos-Predict2 + T5 `llm_adapter` (text adapter kept high-precision) |

## Features

- **GGUF quantization**: q4_0, q4_1, q5_0, q5_1, q8_0, q2_k, q3_k, q4_k, q5_k, q6_k, mxfp4, bf16, f16, f32
- **SafeTensors conversion**: F32, F16, BF16, F8_E4M3, F8_E5M2, SCALED_F8_E4M3, MXFP8_E4M3, NVFP4, INT8, INT8_CONVROT, INT4_CONVROT, INT4_CONVROT_SR, ASYM_W4A8_INT8, MXFP4
- **GGUF <-> SafeTensors roundtrip** — convert between formats without re-quantizing
- **Sensitivity-aware quantization** — per-tensor quantization type selection based on architecture sensitivity data
- **ComfyUI cluster support** — SCALED_F8_E4M3, NVFP4, INT8, INT8_CONVROT, INT4_CONVROT, MXFP4, MXFP8_E4M3, ASYM_W4A8_INT8, INT4_CONVROT_SR
- **Sharded safetensors** — reads and writes multi-file checkpoints transparently
- **Architecture detection** — auto-detects model type from tensor keys
- **CLI + GUI** — full-featured command line and tkinter interface

## Anima support

Anima is a 2B text-to-image model built on NVIDIA
[Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image)
(MiniTrainDIT) with a bolted-on T5 text adapter (`llm_adapter`).

Key details:

- Detected via three keys: `blocks.0.mlp.layer1.weight`, `blocks.0.adaln_modulation_cross_attn.1.weight`, and the discriminator `llm_adapter.blocks.0.cross_attn.q_proj.weight`
- Prefixes `net.`, `model.`, and `model.diffusion_model.` are stripped before matching, so both single-file and Hub-layout checkpoints work
- The entire `llm_adapter` (embedding + 6 cross/self-attention blocks) is kept high-precision, matching the reference converter [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- Maps to `general.architecture = "cosmos"` for ComfyUI-GGUF compatibility (override with `-A anima` if needed)

## Requirements

- Python 3.8+
- `numpy`

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install numpy
```

## Quick start

```bash
# Convert a model to GGUF
python main.py convert model.safetensors -d q4_k

# Convert with specific output
python main.py convert model.safetensors -d q8_0 -o ./converted/ -n my-model-q8

# Check output size without writing
python main.py convert model.safetensors -d q4_k -c

# Use template from an existing GGUF
python main.py template existing.gguf
python main.py convert model.safetensors -t template.json
```

## CLI reference

### Commands

| Command | Description |
|---------|-------------|
| `convert <file>` | Convert SafeTensors to GGUF (or vice versa) |
| `template <file>` | Export a template.json from a GGUF file |
| `header <file>` | Print tensor info and sizes |
| `tree <file>` | Print directory tree |
| `metadata <file>` | Print file metadata |
| `names <file>` | Print all tensor names |
| `sensitivities <file>` | Print per-layer sensitivity data |
| `gui` | Launch tkinter GUI |
| `version` | Print version |

### Options

```
-d, --datatype          Target quantization type (default: source datatype)
-f, --filetype          Target file format: gguf, safetensors (default: gguf)
-t, --template          Use a JSON template for conversion
-o, --output-dir        Output directory (default: same as source)
-n, --output-name       Output filename without extension
-j, --threads           Number of threads for quantization (default: CPU core count)
-a, --aggressiveness    Aggressiveness of sensitivity-aware quantization (default: 50)
-x, --skip-sensitivity  Skip sensitivity-aware quantization
-s, --sensitivities     Path to a sensitivities JSON file
-q, --use-quant-types   Quantization families to use with sensitivity (e.g. "k", "0,k")
-m, --model-only        When output is safetensors, convert only the main model
-u, --allow-unknown-arch Allow converting files with unrecognized architectures
-U, --allow-upscale     Allow converting from lower-precision source to higher-precision target
-A, --arch              Set the architecture name written to GGUF metadata
-R, --stochastic-rounding Seed for INT4_CONVROT_SR stochastic rounding
-c, --calculate-size    Compute and print the exact final output size without writing
-S, --shapes            With names: emit {"name":...,"shape":[...]} objects
```

## GUI

```bash
python main.py gui
# or
ggufy-gui
```

The GUI supports all conversion options, file browsing, template editing,
architecture selection, and real-time progress tracking.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Package layout

```
ggufy/
  cli.py              Command-line interface
  convert.py          Conversion pipeline (types, sensitivity, shape fix, writers)
  data_transform.py   FP8/FP4/BF16 conversions, Hadamard rotation, INT8/INT4/W4A8
  ggml.py             GGML block layouts and reference quantize/dequantize
  gguf.py             GGUF reader/writer
  safetensor.py       SafeTensors reader/writer (single + sharded)
  tensor_clusters.py  ComfyUI cluster grouping, dequantization and writing
  image_arch.py       Architecture detection
  file_loader.py      Unified model loader
  gui.py              tkinter GUI
  types.py            Shared data types
  sensitivities/      Built-in sensitivity data
  configs/            Base architecture configs
tests/
  test_anima.py       Anima detection / rule tests
```

## License

MIT — (c) Quentin Skousen, ggufy contributors
