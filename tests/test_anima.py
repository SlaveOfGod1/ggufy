"""Tests for the Anima-only fork.

Verifies that the official Anima checkpoint
(https://huggingface.co/circlestone-labs/Anima) is detected as the `anima`
architecture, that the high-precision/ignore rules behave correctly, and that
non-Anima checkpoints are NOT detected (this fork is Anima-only).

Run from the fork root with numpy available:

    python -m unittest discover -s tests -v
"""

import unittest

from ggufy import image_arch


# Representative tensor names from the official Anima checkpoint. The real file
# uses the `net.` prefix and contains blocks.0..27, llm_adapter.blocks.0..5 and
# the embedder/final-layer tensors (mirrors ggufy/src/test_fixtures/anima.json).
ANIMA_NAMES = [
    "net.blocks.0.mlp.layer1.weight",
    "net.blocks.0.mlp.layer2.weight",
    "net.blocks.0.adaln_modulation_cross_attn.1.weight",
    "net.blocks.0.self_attn.q_proj.weight",
    "net.blocks.1.adaln_modulation_mlp.1.weight",
    "net.blocks.1.cross_attn.output_proj.weight",
    "net.blocks.14.cross_attn.k_norm.weight",
    "net.blocks.27.mlp.layer2.weight",
    "net.final_layer.adaln_modulation.1.weight",
    "net.final_layer.linear.weight",
    "net.llm_adapter.embed.weight",
    "net.llm_adapter.norm.weight",
    "net.llm_adapter.out_proj.weight",
    "net.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    "net.llm_adapter.blocks.0.self_attn.o_proj.weight",
    "net.llm_adapter.blocks.5.mlp.2.weight",
    "net.t_embedder.1.linear_1.weight",
    "net.t_embedding_norm.weight",
    "net.x_embedder.proj.1.weight",
]


class AnimaDetectionTests(unittest.TestCase):
    def test_official_anima_is_detected(self):
        arch = image_arch.detect_arch(ANIMA_NAMES)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.name, "anima")

    def test_model_diffusion_model_prefix_stripped_for_detection(self):
        prefixed = ["model.diffusion_model." + name[4:] for name in ANIMA_NAMES]
        arch = image_arch.detect_arch(prefixed)
        self.assertIsNotNone(arch)
        self.assertEqual(arch.name, "anima")

    def test_cosmos_without_llm_adapter_is_unknown(self):
        names = [n for n in ANIMA_NAMES if "llm_adapter" not in n]
        self.assertIsNone(image_arch.detect_arch(names))

    def test_unrelated_checkpoint_is_unknown(self):
        flux_like = [
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ]
        self.assertIsNone(image_arch.detect_arch(flux_like))


class AnimaRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = image_arch.detect_arch(ANIMA_NAMES)

    def test_llm_adapter_is_high_precision(self):
        for name in (
            "llm_adapter.embed.weight",
            "llm_adapter.blocks.0.cross_attn.q_proj.weight",
            "llm_adapter.blocks.5.mlp.2.weight",
            "llm_adapter.out_proj.weight",
            "llm_adapter.norm.weight",
        ):
            self.assertTrue(self.arch.is_high_precision(name), name)

    def test_block0_and_sensitive_layers_high_precision(self):
        for name in (
            "blocks.0.mlp.layer1.weight",
            "blocks.1.adaln_modulation_mlp.1.weight",
            "final_layer.linear.weight",
            "t_embedder.1.linear_1.weight",
            "x_embedder.proj.1.weight",
        ):
            self.assertTrue(self.arch.is_high_precision(name), name)

    def test_mid_blocks_not_high_precision(self):
        for name in (
            "blocks.5.mlp.layer1.weight",
            "blocks.14.cross_attn.k_proj.weight",
            "blocks.27.mlp.layer2.weight",
        ):
            self.assertFalse(self.arch.is_high_precision(name), name)

    def test_ignore_rules(self):
        self.assertTrue(self.arch.should_ignore("blocks.3._extra_state"))
        self.assertTrue(self.arch.should_ignore("blocks.3.accum_1"))


class PrefixTests(unittest.TestCase):
    def test_strip_prefix(self):
        self.assertEqual(
            image_arch.strip_prefix("net.blocks.0.mlp.layer1.weight"),
            "blocks.0.mlp.layer1.weight",
        )
        self.assertEqual(
            image_arch.strip_prefix("model.blocks.0.mlp.layer1.weight"),
            "blocks.0.mlp.layer1.weight",
        )
        self.assertEqual(
            image_arch.strip_prefix("model.diffusion_model.blocks.0.mlp.layer1.weight"),
            "blocks.0.mlp.layer1.weight",
        )
        self.assertEqual(
            image_arch.strip_prefix("blocks.0.mlp.layer1.weight"),
            "blocks.0.mlp.layer1.weight",
        )


if __name__ == "__main__":
    unittest.main()
