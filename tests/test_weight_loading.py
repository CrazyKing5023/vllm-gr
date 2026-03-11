#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Tests for DLRM_v3 weight loading and checkpoint conversion.

Covers:
- Checkpoint conversion (DLRM_v3 -> safetensors with hstu. prefix)
- Weight loading via load_weights (parameters + buffers)
- Key naming consistency between DlrmHSTU and HSTUForRecommendation
- HSTUConfig dtype defaults
"""

import os

import pytest
import torch


@pytest.fixture
def hstu_config():
    """Minimal HSTUConfig for testing."""
    from vllm_gr.config.hstu_hf_config import HSTUConfig

    return HSTUConfig(
        max_seq_len=64,
        max_num_candidates=5,
        max_num_candidates_inference=5,
        hstu_num_heads=2,
        hstu_attn_linear_dim=32,
        hstu_attn_qk_dim=16,
        hstu_attn_num_layers=2,
        hstu_embedding_table_dim=16,
        hstu_preprocessor_hidden_dim=32,
        hstu_transducer_embedding_dim=32,
        hash_size=100,
        task_names=["is_click"],
        embedding_tables=[
            {
                "name": "item_embedding",
                "embedding_dim": 16,
                "num_embeddings": 100,
                "feature_names": ["item_id", "item_candidate_id"],
            },
        ],
        user_embedding_feature_names=["item_id"],
        item_embedding_feature_names=["item_candidate_id"],
        uih_post_id_feature_name="item_id",
        uih_action_time_feature_name="action_time",
        uih_weight_feature_name="weight",
        hstu_uih_feature_names=["item_id", "action_time", "weight"],
        hstu_candidate_feature_names=[
            "item_candidate_id", "item_query_time",
            "item_action_weights",
        ],
        merge_uih_candidate_feature_mapping=[
            ["item_id", "item_candidate_id"],
            ["action_time", "item_query_time"],
            ["weight", "item_action_weights"],
        ],
        is_inference=True,
    )


@pytest.fixture
def model_and_config(hstu_config):
    """Create a model instance and its config."""
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    model = HSTUForRecommendation(config=hstu_config)
    return model, hstu_config


def _make_fake_dense_checkpoint(model, tmp_path):
    """Create a fake DLRM_v3 dense checkpoint from model weights."""
    hstu = model.hstu
    dense_dict = {}
    for name, param in hstu.named_parameters():
        if "embedding_collection" not in name:
            dense_dict[name] = param.data.clone()
    for name, buf in hstu.named_buffers():
        if "embedding_collection" not in name:
            dense_dict[name] = buf.clone()

    ckpt_dir = tmp_path / "gr_ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(
        {"dense_dict": dense_dict, "optimizer_dict": {},
         "class_metrics": {"train": [], "eval": []},
         "reg_metrics": {"train": [], "eval": []},
         "global_step": 0, "sparse_tensor_keys": set()},
        str(ckpt_dir / "non_sparse.ckpt"),
    )
    return ckpt_dir, dense_dict


# ── Checkpoint Conversion ──────────────────────────────────────────


class TestCheckpointConversion:

    def test_convert_adds_hstu_prefix(self, model_and_config, tmp_path):
        """Converted keys = 'hstu.' + original dense keys (params + buffers)."""
        model, config = model_and_config
        ckpt_dir, _ = _make_fake_dense_checkpoint(model, tmp_path)

        from scripts.convert_gr_checkpoint import load_dense_weights

        loaded = load_dense_weights(str(ckpt_dir))
        for key in loaded:
            assert not key.startswith("hstu."), (
                f"Raw loaded key should NOT have hstu. prefix: {key}"
            )

        converted = {"hstu." + k: v for k, v in loaded.items()}

        model_keys = {
            name for name, _ in model.named_parameters()
            if "embedding_collection" not in name
        }
        model_keys |= {
            name for name, _ in model.named_buffers()
            if "embedding_collection" not in name
        }

        assert model_keys == set(converted.keys()), (
            f"Key mismatch.\n"
            f"  In model but not converted: {model_keys - set(converted.keys())}\n"
            f"  In converted but not model: {set(converted.keys()) - model_keys}"
        )

    def test_convert_casts_to_bfloat16(self, model_and_config, tmp_path):
        """Conversion casts all tensors to bfloat16."""
        model, config = model_and_config
        ckpt_dir, _ = _make_fake_dense_checkpoint(model, tmp_path)

        config.save_pretrained(str(tmp_path / "hf_model"))

        from scripts.convert_gr_checkpoint import convert

        output_dir = str(tmp_path / "hf_model")
        convert(
            input_path=str(ckpt_dir),
            output_dir=output_dir,
            dense_only=True,
            output_format="pt",
            target_dtype="bfloat16",
        )

        loaded = torch.load(
            os.path.join(output_dir, "model.pt"), weights_only=False
        )
        for k, v in loaded.items():
            assert k.startswith("hstu.")
            assert v.dtype == torch.bfloat16, (
                f"Tensor {k} dtype={v.dtype}, expected bfloat16"
            )


# ── Weight Loading ─────────────────────────────────────────────────


class TestWeightLoading:

    def test_load_weights_restores_params(self, model_and_config):
        """load_weights restores parameter values from a state dict."""
        model, _ = model_and_config

        target_state = {}
        for name, param in model.named_parameters():
            t = torch.zeros_like(param.data)
            t.fill_(0.42)
            target_state[name] = t

        loaded = model.load_weights(target_state.items())
        assert len(loaded) > 0, "No weights were loaded"

        for name, param in model.named_parameters():
            if name in loaded:
                torch.testing.assert_close(param.data, target_state[name])

    def test_load_weights_returns_set(self, model_and_config):
        """load_weights returns a set of loaded key names."""
        model, _ = model_and_config

        weights = [
            (name, param.data.clone())
            for name, param in model.named_parameters()
        ]
        loaded = model.load_weights(iter(weights))

        assert isinstance(loaded, set)
        assert len(loaded) > 0


# ── Key Mapping ────────────────────────────────────────────────────


class TestKeyMapping:

    def test_wrapper_adds_hstu_prefix(self, model_and_config):
        """HSTUForRecommendation prefixes all DlrmHSTU keys with 'hstu.'."""
        model, _ = model_and_config

        wrapper_params = {n for n, _ in model.named_parameters()}
        wrapper_bufs = {n for n, _ in model.named_buffers()}
        wrapper_keys = wrapper_params | wrapper_bufs

        hstu_params = {n for n, _ in model.hstu.named_parameters()}
        hstu_bufs = {n for n, _ in model.hstu.named_buffers()}
        hstu_keys = hstu_params | hstu_bufs

        for key in hstu_keys:
            assert ("hstu." + key) in wrapper_keys, (
                f"Expected hstu.{key} in wrapper"
            )


# ── Config Defaults ────────────────────────────────────────────────


class TestHSTUConfig:

    def test_defaults_to_bfloat16(self):
        from vllm_gr.config.hstu_hf_config import HSTUConfig

        config = HSTUConfig()
        dtype = config.torch_dtype
        assert dtype == "bfloat16" or dtype == torch.bfloat16

    def test_respects_explicit_dtype(self):
        from vllm_gr.config.hstu_hf_config import HSTUConfig

        config = HSTUConfig(torch_dtype="float16")
        assert config.torch_dtype == torch.float16


# ── GREngineArgs ───────────────────────────────────────────────────


class TestArgUtils:

    def test_has_model_weights_safetensors(self, tmp_path):
        from vllm_gr.engine.arg_utils import GREngineArgs

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        (hf_dir / "model.safetensors").touch()

        args = GREngineArgs.__new__(GREngineArgs)
        args.model = str(hf_dir)
        assert args._has_model_weights()

    def test_has_model_weights_empty_dir(self, tmp_path):
        from vllm_gr.engine.arg_utils import GREngineArgs

        empty = tmp_path / "empty"
        empty.mkdir()

        args = GREngineArgs.__new__(GREngineArgs)
        args.model = str(empty)
        assert not args._has_model_weights()
