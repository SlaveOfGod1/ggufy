"""Model architecture detection.

Port of src/ImageArch.zig: detection key sets, high-precision key patterns,
ignored keys, shape rules and upcast lists. Detection runs over the tensor
list (stripping prefixes).

This fork is dedicated to the official Anima model
(https://huggingface.co/circlestone-labs/Anima): a 2B text-to-image
diffusion model built on NVIDIA Cosmos-Predict2 with a bolted-on T5 text
adapter (`llm_adapter`). Only the `anima` architecture is registered in
ARCH_LIST, so any other model is reported as an unknown architecture.
"""

from __future__ import annotations

from typing import List, Optional


class ShapeRule:
    def __init__(self, key: str, dim: int, extent: int):
        self.key = key
        self.dim = dim
        self.extent = extent


class Arch:
    def __init__(self, name, shape_fix=False, keys_detect=None, keys_banned=None,
                 shape_detect=None, keys_hiprec=None, keys_ignore=None,
                 threshold=None, sensitivities="", upcast_from_bf16=None,
                 keys_nvfp4_passthrough=None, base_config_json=""):
        self.name = name
        self.shape_fix = shape_fix
        self.keys_detect = keys_detect or []
        self.keys_banned = keys_banned or []
        self.shape_detect = shape_detect or []
        self.keys_hiprec = keys_hiprec or []
        self.keys_ignore = keys_ignore or []
        self.threshhold = threshold
        self.sensitivities = sensitivities
        self.upcast_from_bf16 = upcast_from_bf16 or []
        self.keys_nvfp4_passthrough = keys_nvfp4_passthrough or []
        self.base_config_json = base_config_json

    def matches(self, tensor_names) -> bool:
        for key_set in self.keys_detect:
            if all_keys_present(key_set, tensor_names):
                banned = False
                for banned_key in self.keys_banned:
                    if contains_key(tensor_names, banned_key):
                        banned = True
                        break
                if banned:
                    continue
                return True
        return False

    def shapes_match(self, tensors) -> bool:
        for rule in self.shape_detect:
            t = find_tensor(tensors, rule.key)
            if t is None:
                return False
            if rule.dim >= len(t.dims):
                return False
            if t.dims[rule.dim] != rule.extent:
                return False
        return True

    def is_high_precision(self, key: str) -> bool:
        for hiprec in self.keys_hiprec:
            if hiprec in key:
                return True
        return False

    def should_ignore(self, key: str) -> bool:
        for ignore in self.keys_ignore:
            if ignore in key:
                return True
        return False

    def find_banned_key(self, tensor_names) -> Optional[str]:
        for banned in self.keys_banned:
            if contains_key(tensor_names, banned):
                return banned
        return None

    def has_banned_keys(self, tensor_names) -> bool:
        return self.find_banned_key(tensor_names) is not None

    def is_nvfp4_passthrough(self, key: str) -> bool:
        for pattern in self.keys_nvfp4_passthrough:
            if pattern in key:
                return True
        return False

    def should_upcast(self, tensor_name: str) -> bool:
        for pattern in self.upcast_from_bf16:
            if pattern and pattern[0] == '.':
                if tensor_name.endswith(pattern):
                    return True
            else:
                if tensor_name == pattern:
                    return True
        return False


def all_keys_present(key_set, tensor_names) -> bool:
    for key in key_set:
        if not contains_key(tensor_names, key):
            return False
    return True


def contains_key(tensor_names, key: str) -> bool:
    for name in tensor_names:
        if strip_prefix(name) == key:
            return True
    return False


def find_tensor(tensors, key: str):
    for t in tensors:
        if strip_prefix(t.name) == key:
            return t
    return None


def find_banned_key_in_tensors(arch: Arch, tensors) -> Optional[str]:
    return arch.find_banned_key([t.name for t in tensors[:4096]])


def has_banned_keys_in_tensors(arch: Arch, tensors) -> bool:
    return find_banned_key_in_tensors(arch, tensors) is not None


# ---------------------------------------------------------------------------
# Architecture definition (from src/ImageArch.zig)
# ---------------------------------------------------------------------------

# Anima is Cosmos-Predict2 (MiniTrainDIT) with an extra bolted-on T5 text
# adapter (`llm_adapter`). It shares Cosmos's entire backbone, so its detect
# keys are Cosmos's two plus the llm_adapter discriminator. This mirrors
# ComfyUI's own model_detection.py, which starts at "cosmos_predict2" and
# reclassifies to "anima" iff `llm_adapter.blocks.0.cross_attn.q_proj.weight`
# is present. "anima" is a valid `general.architecture` value for the
# ComfyUI-GGUF loader (it's in PIG_ARCH_LIST), so we can name it distinctly.
#
# The ENTIRE `llm_adapter` is kept high-precision (not just its embedding),
# matching the reference converter silveroxides/convert_to_quant (its
# ANIMA_LAYER_KEYNAMES lists "llm_adapter" as highprec). Two reasons:
#   1. ComfyUI: the adapter's `embed.weight` is an nn.Embedding table that
#      can't be block/int-quantized (also caught generically by
#      is_embedding_weight() in convert.py).
#   2. Forge Neo: its loader's `process_anima` MOVES the whole llm_adapter out
#      of the transformer and into the *text-encoder* component. If any adapter
#      tensor is quantized, the text encoder loads via its MixedPrecision path
#      and scaled_dot_product_attention throws on mismatched dtypes. Keeping
#      the adapter fully bf16 avoids the quantized path entirely.
#
# High-precision set also includes the first block, block 1's adaln modulation,
# the final layer, and the timestep/patch embedders -- the small,
# sensitivity-critical layers that the reference tool keeps in full precision.
_ANIMA = Arch(
    name="anima",
    keys_detect=[["blocks.0.mlp.layer1.weight",
                  "blocks.0.adaln_modulation_cross_attn.1.weight",
                  "llm_adapter.blocks.0.cross_attn.q_proj.weight"]],
    keys_hiprec=["pos_embedder", "llm_adapter", "blocks.0.",
                 "blocks.1.adaln_modulation", "final_layer", "t_embedder",
                 "x_embedder"],
    keys_ignore=["_extra_state", "accum_"],
)

# Anima-only fork: the official model is the only registered architecture.
# Other image-diffusion architectures are intentionally not listed, so
# detection/conversion targets the official Anima checkpoint exclusively.
# (The generic conversion machinery still works; unknown models simply require
# --allow-unknown-arch like any unrecognized file.)
ARCH_LIST = [_ANIMA]

GENERIC_ARCH = Arch(name="unknown", keys_detect=[])


def strip_prefix(name: str) -> str:
    mixed_prefixes = ["model.diffusion_model.", "model."]
    uniform_prefixes = ["net."]
    for prefix in mixed_prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    for prefix in uniform_prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def arch_matches(arch: Arch, names, tensors) -> bool:
    if not arch.matches(names):
        return False
    if not arch.shape_detect:
        return True
    if tensors is None:
        return False
    return arch.shapes_match(tensors)


def detect_impl(names, tensors) -> Optional[Arch]:
    for arch in ARCH_LIST:
        if arch_matches(arch, names, tensors):
            return arch
    return None


def detect_arch(tensor_names) -> Optional[Arch]:
    return detect_impl(tensor_names, None)


def detect_arch_from_tensors(tensors) -> Optional[Arch]:
    names = [t.name for t in tensors]
    return detect_impl(names, tensors)


def detect_arch_from_tensors_or_error(tensors) -> Arch:
    arch = detect_arch_from_tensors(tensors)
    if arch is None:
        raise ValueError("UnknownArchitecture")
    return arch


class ArchError(Exception):
    pass
