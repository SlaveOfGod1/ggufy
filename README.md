# GGUFy (Python)

A lightweight, robust, pure-Python tool to convert model tensor formats
(SafeTensors ↔ GGUF), designed for seamless use with
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

This is a faithful Python port of [ggufy](https://github.com/qskousen/ggufy)
by `qskousen`, keeping the same CLI, commands, options and on-disk formats as
the original Zig tool. The only runtime dependency is `numpy`.

## Features

- **Real GGML quantization**: `f32`, `bf16`, `f16`, `q8_0`, `q5_0`, `q5_1`,
  `q4_0`, `q4_1`, `q6_k`, `q5_k`, `q4_k`, `q3_k`, `q2_k`, `mxfp4` — block
  layouts match the ggml reference byte-for-byte.
- **SafeTensors datatype conversion**: `F32`, `F16`, `BF16`, `F8_E4M3`,
  `F8_E5M2`, `SCALED_F8_E4M3`, `MXFP8_E4M3`, `NVFP4`, `INT8`, `INT8_CONVROT`,
  `INT4_CONVROT`, `INT4_CONVROT_SR`, `ASYM_W4A8_INT8`, `MXFP4` (ComfyUI
  cluster formats).
- **Sensitivity-aware quantization** for supported architectures (SD1.5,
  SDXL), with built-in sensitivity data.
- **Architecture detection**: flux, sd3, aura, hidream, **anima**, cosmos, ltx2/3,
  ltxv, hyvid, wan, sdxl, sd1, lumina2, mage_flow, qwen, ernie, krea2.
- **Shape fix** with `comfy.gguf.orig_shape` metadata for ComfyUI.
- **Native BF16/FP8/FP4 reading** — pure Python + NumPy, no PyTorch needed.
- **CLI + tkinter GUI**, mirroring the original app.
- Sharded SafeTensors (`model.safetensors.index.json`) support.

## Anima support

This repo also includes support for the official
[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) model
(a 2B text-to-image model built on NVIDIA
[Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image)
with a bolted-on T5 text adapter, `llm_adapter`).

- Detected via `blocks.0.mlp.layer1.weight`, `blocks.0.adaln_modulation_cross_attn.1.weight`,
  and the discriminator `llm_adapter.blocks.0.cross_attn.q_proj.weight`
  (mirrors ComfyUI's `model_detection.py` reclassification of `cosmos_predict2`).
- The entire `llm_adapter` is kept high-precision (its `embed.weight` is an
  `nn.Embedding` that can't be block-quantized), matching the reference
  converter [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant).
- Maps to `general.architecture = "cosmos"` for ComfyUI-GGUF compatibility
  (override with `-A anima` if needed).

## Requirements

- Python 3.8+
- `numpy` (install into your venv manually if you prefer)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install numpy
```

## CLI usage

```bash
# Run from this directory
python main.py convert model.safetensors -d q4_k

# Or install as a console script
pip install .
ggufy convert model.safetensors -d q4_k
```

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

## Package layout

```
ggufy/
  cli.py            Command-line interface
  convert.py        Conversion pipeline (types, sensitivity, shape fix, writers)
  data_transform.py FP8/FP4/BF16 conversions, Hadamard rotation, INT8/INT4/W4A8
  ggml.py           GGML block layouts and reference quantize/dequantize
  gguf.py           GGUF reader/writer
  safetensor.py     SafeTensors reader/writer (single + sharded)
  tensor_clusters.py ComfyUI cluster grouping, dequantization and writing
  image_arch.py     Architecture detection
  file_loader.py    Unified model loader
  gui.py            tkinter GUI
  types.py          Shared data types
  sensitivities/    Built-in sensitivity data (sd1.5, sdxl, ...)
  configs/          Base architecture configs (ltx2/3)
```

## Acknowledgements

- [ggufy (Zig)](https://github.com/qskousen/ggufy) — the original tool this
  port is based on.
- [ggml](https://github.com/ggml-org/ggml) — quantization algorithms ported
  from the reference implementation.
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) — for documenting the
  quantization and architecture detection formats.
