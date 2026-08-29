# ggufy (Python) — Anima edition

A Python port of [ggufy](https://github.com/qskousen/ggufy): a lightweight and
efficient tool to convert tensor formats. This **fork is dedicated to the
official Anima model only** — [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima).

Anima is a 2B text-to-image model built on NVIDIA
[Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image)
(MiniTrainDIT) with a bolted-on T5 text adapter (`llm_adapter`). ComfyUI
and Forge-Neo load it as `general.architecture = "anima"` via their GGUF
paths.

Scope

- The original Python port (`../python version/`) supports many architectures (flux, sd3, aura, hidream, ltxv, sdxl, lumina2, qwen, krea2, etc.). Detection was restricted here to the official Anima checkpoint — only `anima` is registered in `ARCH_LIST`. Anything else reports `UnknownArchitecture` unless `--allow-unknown-arch` is passed.
- Detection uses three keys: `blocks.0.mlp.layer1.weight`,
  `blocks.0.adaln_modulation_cross_attn.1.weight`, and the Anima
  discriminator `llm_adapter.blocks.0.cross_attn.q_proj.weight` (mirrors
  ComfyUI's `model_detection.py` reclassification of `cosmos_predict2`).
  Prefixes `net.`, `model.`, and `model.diffusion_model.` are stripped
  before matching, so both single-file and Hub-layout checkpoints work.
- The entire `llm_adapter` (embedding + 6 cross/self-attention blocks) is
  kept high-precision (`pos_embedder`, `llm_adapter`, `blocks.0.`,
  `blocks.1.adaln_modulation`, `final_layer`, `t_embedder`, `x_embedder`).
  This matches the reference converter
  [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
  (`ANIMA_LAYER_KEYNAMES`): the adapter's `embed.weight` is an
  `nn.Embedding` (cannot be block-quantized), and Forge-Neo's
  `process_anima` moves the adapter into the text-encoder component — any
  quantized projection would force the encoder through its `MixedPrecision`
  path and throw on mismatched `bf16`/`f32` rope-vs-v dtypes.

Differences from the upstream Python port: `ggufy/image_arch.py:14` (module
docstring), `ggufy/image_arch.py:132` (single `_ANIMA` arch + `ARCH_LIST = [_ANIMA]`; other archs removed),
`tests/test_anima.py` (added), `pyproject.toml:8` (description), and this README.

## Requirements

- Python 3.8+
- `numpy` (install into your venv manually if you prefer):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install numpy
```

## CLI usage (Anima)

Place the official files where ComfyUI expects them, or convert from any location
with this tool.

- `anima-base-v1.0.safetensors` (diffusion model, single-file) goes in `ComfyUI/models/diffusion_models`
- `qwen_3_06b_base.safetensors` goes in `ComfyUI/models/text_encoders`
- `qwen_image_vae.safetensors` goes in `ComfyUI/models/vae` (the Qwen-Image VAE)

Convert Anima to GGUF (pick one datatype):

```bash
# Run from this directory
python main.py convert anima-base-v1.0.safetensors -d q4_k
python main.py convert anima-base-v1.0.safetensors -d q8_0 -n anima-base-q8_0 -o ./converted/

# Or install as a console script
pip install .
ggufy convert anima-base-v1.0.safetensors -d q4_k
```

The CLI mirrors the original:

```bash
ggufy convert anima-base-v1.0.safetensors -d q4_k -n anima-q4_k -o ./converted/
ggufy template existing.gguf                          # export template.json
ggufy convert anima-base-v1.0.safetensors -t template.json      # convert using template
ggufy header anima-base-v1.0.safetensors
ggufy tree anima-base-v1.0.safetensors
ggufy metadata model.gguf
ggufy names anima-base-v1.0.safetensors
ggufy sensitivities anima-base-v1.0.safetensors
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

Available output types (GGUF): `f32`, `bf16`, `f16`, `q8_0`, `q5_0`, `q5_1`,
`q4_0`, `q4_1`, `q6_k`, `q5_k`, `q4_k`, `q3_k`, `q2_k`, `mxfp4`.

Available output types (SafeTensors): `F32`, `F16`, `BF16`, `F8_E4M3`,
`F8_E5M2`, `SCALED_F8_E4M3`, `MXFP8_E4M3`, `NVFP4`, `INT8`, `INT8_CONVROT`,
`INT4_CONVROT`, `INT4_CONVROT_SR`, `ASYM_W4A8_INT8`, `MXFP4`.

Sharded layouts (`model.safetensors.index.json`) are handled transparently.

## GUI

A tkinter GUI (no extra dependencies) mirrors the original app:

```bash
python main.py gui
# or
ggufy-gui
```

## Tests (Anima detection)

```bash
# requires numpy to be installed (same as the converter itself)
python -m unittest discover -s tests -v
```

Checks that the official names (`net.blocks.*`, `net.llm_adapter.*`,
`net.t_embedder.*`, etc.) round-trip through prefix stripping, that the
cosmos backbone without `llm_adapter` is not detected (fork is Anima-only),
and that the hiprec/ignore rules match the Zig reference fixture
`ggufy/src/test_fixtures/anima.json`.

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
  image_arch.py     Architecture detection — Anima only in this fork
  file_loader.py    Unified model loader
  gui.py            tkinter GUI
  types.py          Shared data types
  sensitivities/    Built-in sensitivity data (kept; not used by Anima)
  configs/          Base architecture configs (kept; not used by Anima)
tests/
  test_anima.py     Anima detection / rule tests (added in this fork)
```
