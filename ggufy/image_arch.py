"""Model architecture detection.

Port of src/ImageArch.zig: a list of known image-diffusion architectures with
detection key sets, high-precision key patterns, ignored keys, shape rules and
upcast lists. Detection runs over the tensor list (stripping prefixes), and
architectures that share a tensor-name set (Mage-Flow vs Qwen-Image) are
disambiguated by shape rules.
"""

from __future__ import annotations

import os
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
# Architecture definitions (from src/ImageArch.zig)
# ---------------------------------------------------------------------------

_FLUX = Arch(
    name="flux", shape_fix=True,
    keys_detect=[["transformer_blocks.0.attn.norm_added_k.weight"],
                 ["double_blocks.0.img_attn.proj.weight"]],
    keys_banned=["transformer_blocks.0.attn.norm_added_k.weight"],
    upcast_from_bf16=[".norm.query_norm.scale", ".norm.key_norm.scale",
                      ".norm.query_norm.weight", ".norm.key_norm.weight"],
    keys_nvfp4_passthrough=["img_in.weight", "txt_in.weight",
                            "vector_in.in_layer.weight"],
)

_SD3 = Arch(
    name="sd3",
    keys_detect=[["transformer_blocks.0.attn.add_q_proj.weight"],
                 ["joint_blocks.0.x_block.attn.qkv.weight"]],
    keys_banned=["transformer_blocks.0.attn.add_q_proj.weight"],
    keys_nvfp4_passthrough=["y_embedder.mlp.0.weight", "context_embedder.weight"],
)

_AURA = Arch(
    name="aura",
    keys_detect=[["double_layers.3.modX.1.weight"],
                 ["joint_transformer_blocks.3.ff_context.out_projection.weight"]],
    keys_banned=["joint_transformer_blocks.3.ff_context.out_projection.weight"],
)

_HIDREAM = Arch(
    name="hidream",
    keys_detect=[["caption_projection.0.linear.weight",
                  "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"]],
    keys_hiprec=[".ff_i.gate.weight", "img_emb.emb_pos"],
)

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

_COSMOS = Arch(
    name="cosmos",
    keys_detect=[["blocks.0.mlp.layer1.weight",
                  "blocks.0.adaln_modulation_cross_attn.1.weight"]],
    keys_hiprec=["pos_embedder"],
    keys_ignore=["_extra_state", "accum_"],
)

_HYVID = Arch(
    name="hyvid",
    keys_detect=[["double_blocks.0.img_attn_proj.weight",
                  "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"]],
)

_WAN = Arch(
    name="wan",
    keys_detect=[["blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight",
                  "head.modulation"]],
    keys_hiprec=[".modulation"],
)

_LTXV = Arch(
    name="ltxv",
    keys_detect=[["adaln_single.emb.timestep_embedder.linear_2.weight",
                  "transformer_blocks.27.scale_shift_table",
                  "caption_projection.linear_2.weight"]],
    keys_hiprec=["scale_shift_table"],
)

with open(os.path.join(os.path.dirname(__file__), "configs", "ltx23_base_config.json"),
          "r", encoding="utf-8") as _f:
    _LTX23_BASE = _f.read()

_LTX2 = Arch(
    name="ltxv",
    base_config_json=_LTX23_BASE,
    keys_detect=[["adaln_single.emb.timestep_embedder.linear_2.weight",
                  "transformer_blocks.47.scale_shift_table",
                  "patchify_proj.weight"]],
    keys_hiprec=["scale_shift_table", "_norm.weight", ".bias", "adaln_single",
                 "patchify_proj.weight", "proj_out.weight", "learnable_registers"],
)

with open(os.path.join(os.path.dirname(__file__), "sensitivities", "sdxl.json"),
          "r", encoding="utf-8") as _f:
    _SDXL_SENS = _f.read()

_SDXL = Arch(
    name="sdxl", shape_fix=True,
    keys_detect=[["down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight"],
                 ["input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
                  "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"],
                 ["label_emb.0.0.weight"]],
    sensitivities=_SDXL_SENS,
    keys_nvfp4_passthrough=["label_emb.0.0.weight"],
)

with open(os.path.join(os.path.dirname(__file__), "sensitivities", "sd1.5.json"),
          "r", encoding="utf-8") as _f:
    _SD1_SENS = _f.read()

_SD1 = Arch(
    name="sd1", shape_fix=True,
    keys_detect=[["down_blocks.0.downsamplers.0.conv.weight"],
                 ["input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
                  "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight",
                  "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"]],
    sensitivities=_SD1_SENS,
    keys_nvfp4_passthrough=["label_emb.0.0.weight"],
)

_LUMINA2 = Arch(
    name="lumina2",
    keys_detect=[["cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight"]],
    shape_fix=True,
    keys_ignore=["norm_final.weight"],
    threshold=8192,
    upcast_from_bf16=["cap_pad_token", "x_pad_token"],
    keys_nvfp4_passthrough=["cap_embedder.1.weight"],
)

_QWEN = Arch(
    name="qwen",
    keys_detect=[["time_text_embed.timestep_embedder.linear_2.weight",
                  "transformer_blocks.0.attn.norm_added_q.weight",
                  "transformer_blocks.0.img_mlp.net.0.proj.weight"]],
    shape_fix=True,
    upcast_from_bf16=["txt_norm.weight", ".norm_k.weight", ".norm_q.weight",
                      ".norm_added_k.weight", ".norm_added_q.weight"],
    keys_nvfp4_passthrough=["img_in.weight"],
)

_MAGEFLOW = Arch(
    name="mage_flow",
    keys_detect=[["time_text_embed.timestep_embedder.linear_2.weight",
                  "transformer_blocks.0.attn.norm_added_q.weight",
                  "transformer_blocks.0.img_mlp.net.0.proj.weight",
                  "txt_norm.weight", "proj_out.weight"]],
    shape_detect=[ShapeRule("txt_norm.weight", 0, 2560),
                  ShapeRule("proj_out.weight", 0, 128)],
    shape_fix=True,
    keys_hiprec=["txt_norm.weight", "img_in.", "txt_in.", "proj_out.",
                 "norm_out.linear", "time_text_embed"],
    upcast_from_bf16=["txt_norm.weight", ".norm_k.weight", ".norm_q.weight",
                      ".norm_added_k.weight", ".norm_added_q.weight"],
)

_ERNIE = Arch(
    name="ernie",
    keys_detect=[["adaLN_modulation.1.weight", "x_embedder.proj.weight",
                  "text_proj.weight", "layers.0.mlp.linear_fc2.weight"]],
    shape_fix=True,
    upcast_from_bf16=[".adaLN_sa_ln.weight", ".adaLN_mlp_ln.weight"],
)

_KREA2 = Arch(
    name="krea2",
    keys_detect=[["blocks.0.attn.qknorm.qnorm.scale", "txtfusion.projector.weight"]],
    shape_fix=True,
    keys_hiprec=["txtfusion", "tmlp", "tproj", "first.", "last.", ".projector"],
    upcast_from_bf16=[".qknorm.qnorm.scale", ".qknorm.knorm.scale",
                      ".prenorm.scale", ".postnorm.scale"],
    keys_nvfp4_passthrough=["first.weight"],
)

ARCH_LIST = [_FLUX, _SD3, _AURA, _HIDREAM, _ANIMA, _COSMOS, _LTX2, _LTXV,
             _HYVID, _WAN, _SDXL, _SD1, _LUMINA2, _MAGEFLOW, _QWEN, _ERNIE,
             _KREA2]

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
