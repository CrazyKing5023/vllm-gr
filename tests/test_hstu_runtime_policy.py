# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Lightweight tests for HSTU runtime device policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_fake_hstu_config(**overrides):
    base = {
        "max_seq_len": 64,
        "max_num_candidates": 8,
        "max_num_candidates_inference": 8,
        "hstu_num_heads": 2,
        "hstu_attn_linear_dim": 32,
        "hstu_attn_qk_dim": 16,
        "hstu_attn_num_layers": 2,
        "hstu_embedding_table_dim": 16,
        "hstu_preprocessor_hidden_dim": 32,
        "hstu_transducer_embedding_dim": 32,
        "hstu_group_norm": False,
        "hstu_input_dropout_ratio": 0.0,
        "hstu_linear_dropout_rate": 0.0,
        "contextual_feature_to_max_length": {},
        "contextual_feature_to_min_uih_length": {},
        "candidates_weight_feature_name": "",
        "candidates_watchtime_feature_name": "",
        "candidates_querytime_feature_name": "",
        "task_names": ["is_click"],
        "user_embedding_feature_names": ["item_id"],
        "item_embedding_feature_names": ["item_candidate_id"],
        "uih_post_id_feature_name": "item_id",
        "uih_action_time_feature_name": "",
        "uih_weight_feature_name": "",
        "hstu_uih_feature_names": ["item_id"],
        "hstu_candidate_feature_names": ["item_candidate_id"],
        "merge_uih_candidate_feature_mapping": [],
        "action_weights": [],
        "action_embedding_init_std": 0.1,
        "causal_multitask_weights": 0.2,
        "enable_postprocessor": True,
        "use_layer_norm_postprocessor": False,
        "embedding_tables": [],
        "is_inference": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestHSTURuntimePolicy(unittest.TestCase):

    def test_runtime_devices_default_to_cpu_embedding(self):
        from vllm_gr.utils.hstu_runtime import resolve_hstu_runtime_devices

        module = nn.Linear(4, 4)
        policy = resolve_hstu_runtime_devices(module)

        self.assertEqual(policy.dense_device.type, "cpu")
        self.assertEqual(policy.embedding_device, torch.device("cpu"))

    def test_runtime_devices_prefer_explicit_overrides(self):
        from vllm_gr.utils.hstu_runtime import resolve_hstu_runtime_devices

        module = nn.Linear(4, 4)
        policy = resolve_hstu_runtime_devices(
            module,
            explicit_dense_device=torch.device("cpu"),
            explicit_embedding_device=torch.device("cpu"),
        )

        self.assertEqual(policy.dense_device, torch.device("cpu"))
        self.assertEqual(policy.embedding_device, torch.device("cpu"))

    def test_sparse_transfer_plan_is_explicit(self):
        from vllm_gr.utils.hstu_runtime import resolve_hstu_sparse_transfer_plan

        no_op = resolve_hstu_sparse_transfer_plan(
            source_device=torch.device("cpu"),
            target_device=torch.device("cpu"),
        )
        cross_device = resolve_hstu_sparse_transfer_plan(
            source_device=torch.device("cpu"),
            target_device=torch.device("cuda"),
        )

        self.assertFalse(no_op.requires_transfer)
        self.assertEqual(no_op.source_device, torch.device("cpu"))
        self.assertEqual(no_op.target_device, torch.device("cpu"))
        self.assertTrue(cross_device.requires_transfer)
        self.assertEqual(cross_device.target_device, torch.device("cuda"))

    def test_sparse_output_move_is_noop_when_devices_match(self):
        from vllm_gr.utils.hstu_runtime import move_hstu_sparse_outputs_to_device

        fake_dlrm_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.modules.dlrm_hstu"
        )

        class FakeSequenceEmbedding(tuple):
            __slots__ = ()

            def __new__(cls, lengths, embedding):
                return super().__new__(cls, (lengths, embedding))

            @property
            def lengths(self):
                return self[0]

            @property
            def embedding(self):
                return self[1]

        fake_dlrm_module.SequenceEmbedding = FakeSequenceEmbedding
        seq_embeddings = {
            "item_id": FakeSequenceEmbedding(
                lengths=torch.tensor([2], dtype=torch.int64),
                embedding=torch.randn(2, 4),
            ),
        }
        payload_features = {"uih_offsets": torch.tensor([0, 2], dtype=torch.int64)}
        uih_seq_lengths = torch.tensor([2], dtype=torch.int64)
        num_candidates = torch.tensor([1], dtype=torch.int64)

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.modules.dlrm_hstu": fake_dlrm_module},
        ):
            moved = move_hstu_sparse_outputs_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=torch.device("cpu"),
                source_device=torch.device("cpu"),
            )

        self.assertIs(moved[0], seq_embeddings)
        self.assertIs(moved[1], payload_features)
        self.assertIs(moved[2], uih_seq_lengths)
        self.assertIs(moved[3], num_candidates)

    def test_dlrm_hstu_keeps_embedding_device_as_consumed_input(self):
        dlrm_hstu_source = (
            ROOT / "vllm_gr" / "models" / "dlrm_v3" / "modules" / "dlrm_hstu.py"
        ).read_text()

        self.assertIn("_consume_embedding_device", dlrm_hstu_source)
        self.assertNotIn("self._embedding_device = resolve_hstu_embedding_device", dlrm_hstu_source)

    def test_wrapper_passes_embedding_device_to_dlrm_hstu(self):
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        fake_dlrm_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.modules.dlrm_hstu"
        )
        fake_multitask_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.modules.multitask_module"
        )

        class FakeDlrmHSTUConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeDlrmHSTU(nn.Module):
            def __init__(self, *args, embedding_device=None, **kwargs):
                super().__init__()
                self._embedding_device = torch.device(
                    embedding_device or torch.device("cpu")
                )
                self.dense_weight = nn.Parameter(torch.zeros(1))

        class FakeTaskConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_multitask_module.TaskConfig = FakeTaskConfig
        fake_multitask_module.MultitaskTaskType = SimpleNamespace(
            BINARY_CLASSIFICATION="binary"
        )
        fake_dlrm_module.DlrmHSTU = FakeDlrmHSTU
        fake_dlrm_module.DlrmHSTUConfig = FakeDlrmHSTUConfig

        config = _build_fake_hstu_config(hstu_embedding_device="cpu")

        with mock.patch.dict(
            sys.modules,
            {
                "vllm_gr.models.dlrm_v3.modules.dlrm_hstu": fake_dlrm_module,
                "vllm_gr.models.dlrm_v3.modules.multitask_module": fake_multitask_module,
            },
        ), mock.patch.object(
            HSTUForRecommendation,
            "_build_embedding_configs",
            return_value={},
        ):
            model = HSTUForRecommendation(config=config, device=torch.device("cpu"))

        self.assertEqual(model._embedding_device, torch.device("cpu"))
        self.assertEqual(model.hstu._embedding_device, torch.device("cpu"))

    def test_pooling_predictions_are_split_per_request(self):
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        predictions = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        num_candidates = torch.tensor([2, 3], dtype=torch.int64)

        split_outputs = HSTUForRecommendation._split_pooling_predictions(
            predictions,
            num_candidates,
        )

        self.assertIsInstance(split_outputs, list)
        self.assertEqual(len(split_outputs), 2)
        torch.testing.assert_close(
            split_outputs[0],
            torch.tensor([[0.1, 0.2]]),
        )
        torch.testing.assert_close(
            split_outputs[1],
            torch.tensor([[0.3, 0.4, 0.5]]),
        )

    def test_pooling_predictions_fallback_when_shape_mismatches(self):
        from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

        predictions = torch.tensor([[0.1, 0.2, 0.3]])
        num_candidates = torch.tensor([2, 2], dtype=torch.int64)

        split_outputs = HSTUForRecommendation._split_pooling_predictions(
            predictions,
            num_candidates,
        )

        self.assertTrue(torch.equal(split_outputs, predictions))


if __name__ == "__main__":
    unittest.main()
