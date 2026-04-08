# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

import importlib.machinery
from types import SimpleNamespace
import sys
import types
import unittest
from unittest import mock

import torch

if "torchrec" not in sys.modules:
    fake_torchrec = types.ModuleType("torchrec")
    fake_torchrec_modules = types.ModuleType("torchrec.modules")
    fake_embedding_configs = types.ModuleType("torchrec.modules.embedding_configs")
    fake_embedding_modules = types.ModuleType("torchrec.modules.embedding_modules")
    fake_torchrec_sparse = types.ModuleType("torchrec.sparse")
    fake_jagged_tensor = types.ModuleType("torchrec.sparse.jagged_tensor")

    class _EmbeddingConfig:
        pass

    class _EmbeddingCollection:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class _KeyedJaggedTensor:
        pass

    fake_torchrec.__spec__ = importlib.machinery.ModuleSpec("torchrec", loader=None)
    fake_torchrec_modules.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules", loader=None
    )
    fake_embedding_configs.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules.embedding_configs", loader=None
    )
    fake_embedding_modules.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.modules.embedding_modules", loader=None
    )
    fake_torchrec_sparse.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse", loader=None
    )
    fake_jagged_tensor.__spec__ = importlib.machinery.ModuleSpec(
        "torchrec.sparse.jagged_tensor", loader=None
    )
    fake_embedding_configs.EmbeddingConfig = _EmbeddingConfig
    fake_embedding_modules.EmbeddingCollection = _EmbeddingCollection
    fake_jagged_tensor.KeyedJaggedTensor = _KeyedJaggedTensor
    sys.modules.update(
        {
            "torchrec": fake_torchrec,
            "torchrec.modules": fake_torchrec_modules,
            "torchrec.modules.embedding_configs": fake_embedding_configs,
            "torchrec.modules.embedding_modules": fake_embedding_modules,
            "torchrec.sparse": fake_torchrec_sparse,
            "torchrec.sparse.jagged_tensor": fake_jagged_tensor,
        }
    )

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_gr.engine.arg_utils import GREngineArgs
from vllm_gr.models.dlrm_v3.modules import stu as stu_module
from vllm_gr.models.dlrm_v3.modules.stu import STULayer, STULayerConfig
from vllm_gr.attention.backends.hstu_attn import (
    HSTUAttentionMetadata,
    HSTUAttentionMetadataBuilder,
    _build_hstu_token_positions,
)
from vllm_gr.model_executor.layers.hstu_attention_layer import (
    HSTUKVCacheAttentionLayer,
)
from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation
from vllm_gr.ops.hstu_attention import delta_hstu_mha


def _dummy_vllm_config(block_size: int = 16):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


class TestHSTUVllmNativeKVContract(unittest.TestCase):

    @staticmethod
    def _make_hstu_model_for_metadata_test():
        model = HSTUForRecommendation.__new__(HSTUForRecommendation)
        torch.nn.Module.__init__(model)
        model.enable_kv_cache = True
        model._hstu_kv_cache_layers = [SimpleNamespace(layer_name="layer0")]
        return model

    @staticmethod
    def _make_attention_metadata_for_coverage_test(
        *,
        slot_count: int,
        computed_tokens: int,
        history_len: int = 6,
        contextual_seq_len: int = 5,
        num_targets: int = 6,
    ) -> HSTUAttentionMetadata:
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, slot_count], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, slot_count], dtype=torch.int32),
            seq_lens=torch.tensor([slot_count], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([slot_count], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([computed_tokens], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=slot_count,
            max_query_len=slot_count,
            max_seq_len=slot_count,
            block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
            slot_mapping=torch.arange(slot_count, dtype=torch.int64),
            causal=True,
        )
        return HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["layer0"],
            cache_layers=None,
        )

    @staticmethod
    def _build_native_kv_test_fixture(
        *,
        seq_len: int,
        history_len: int,
        num_targets: int,
        contextual_seq_len: int = 0,
    ) -> tuple[STULayer, HSTUKVCacheAttentionLayer, HSTUAttentionMetadata, torch.Tensor]:
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.eval()

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )
        x = torch.randn(seq_len, 6, dtype=torch.float32)
        return native_layer, cache_layer, metadata, x

    def test_gr_engine_args_expand_max_model_len_to_full_visible_sequence(self) -> None:
        args = GREngineArgs(
            model=None,
            max_seq_len=128,
            max_candidates=256,
            hf_overrides={
                "contextual_feature_to_max_length": {
                    "user_id": 1,
                    "sex": 1,
                    "age_group": 1,
                    "occupation": 1,
                    "zip_code": 1,
                }
            },
            allow_missing_weights=True,
        )

        self.assertEqual(args.max_model_len, 389)
        self.assertIsInstance(args.hf_overrides, dict)
        self.assertEqual(args.hf_overrides["model_max_length"], 389)

    def test_cache_layer_reports_full_attention_spec(self) -> None:
        layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=4,
            head_size=32,
            head_size_v=64,
            dtype="bfloat16",
            vllm_config=None,
            enable_kv_cache=True,
        )

        spec = layer.get_kv_cache_spec(_dummy_vllm_config())

        self.assertIsInstance(spec, FullAttentionSpec)
        self.assertEqual(spec.block_size, 16)
        self.assertEqual(spec.num_kv_heads, 4)
        self.assertEqual(spec.head_size, 32)
        self.assertEqual(spec.head_size_v, 64)
        self.assertEqual(spec.dtype, torch.bfloat16)

    def test_metadata_builder_splits_prefix_and_delta_positions(self) -> None:
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=4,
            head_size_v=8,
            dtype=torch.bfloat16,
        )
        builder = HSTUAttentionMetadataBuilder(
            spec,
            ["hstu_kv_cache_layers.0"],
            _dummy_vllm_config(),
            torch.device("cpu"),
        )
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 5, 8], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 5, 8], dtype=torch.int32),
            seq_lens=torch.tensor([5, 3], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=8,
            max_query_len=5,
            max_seq_len=5,
            block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
            slot_mapping=torch.tensor([10, 11, 12, 13, 14, 20, 21, 22], dtype=torch.int64),
            causal=True,
        )

        builder.set_request_lengths(
            num_targets=torch.tensor([2, 2], dtype=torch.int32),
            history_lengths=torch.tensor([3, 1], dtype=torch.int32),
            contextual_seq_len=0,
        )
        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

        prefix_positions, delta_positions, full_positions = _build_hstu_token_positions(
            metadata,
            device=torch.device("cpu"),
        )

        self.assertEqual(prefix_positions.tolist(), [0, 1, 2, 5])
        self.assertEqual(delta_positions.tolist(), [3, 4, 6, 7])
        self.assertEqual(full_positions.tolist(), list(range(8)))
        self.assertEqual(
            metadata.get_slot_mapping("prefix", device=torch.device("cpu")).tolist(),
            [10, 11, 12, 20],
        )
        self.assertEqual(
            metadata.get_slot_mapping("delta", device=torch.device("cpu")).tolist(),
            [13, 14, 21, 22],
        )

    def test_attention_impl_delta_path_matches_full_delta_attention(self) -> None:
        layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=1,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        kv_cache = torch.zeros((1, 4, 1, 5), dtype=torch.float32)
        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=3,
            max_query_len=3,
            max_seq_len=3,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([1], dtype=torch.int32),
            history_lengths=torch.tensor([2], dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[layer],
        )

        prefix_k = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float32)
        prefix_v = torch.tensor(
            [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            dtype=torch.float32,
        )
        delta_q = torch.tensor([[[0.5, 1.5]]], dtype=torch.float32)
        delta_k = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
        delta_v = torch.tensor([[[7.0, 8.0, 9.0]]], dtype=torch.float32)

        layer.write_to_paged_cache(
            prefix_k,
            prefix_v,
            common.slot_mapping[:2],
            kv_cache=kv_cache,
        )
        output = layer(
            query=delta_q,
            key=delta_k,
            value=delta_v,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            attn_mode="delta",
        )

        expected_k = torch.cat([prefix_k, delta_k], dim=0)
        expected_v = torch.cat([prefix_v, delta_v], dim=0)
        expected = delta_hstu_mha(
            max_seq_len=3,
            alpha=layer.scale,
            delta_q=delta_q,
            k=expected_k,
            v=expected_v,
            seq_offsets=torch.tensor([0, 3], dtype=torch.int64),
            num_targets=metadata.num_targets,
            contextual_seq_len=0,
        )

        torch.testing.assert_close(output, expected)

    def test_prefix_mode_matches_full_forward_prefix_outputs(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_len = 7
        history_len = 4
        num_targets = 3
        x = torch.randn(seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor([seq_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, seq_len], dtype=torch.int64),
            max_seq_len=seq_len,
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_output = native_layer(
            x=x[:history_len],
            x_lengths=torch.tensor([history_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
            max_seq_len=history_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )

        torch.testing.assert_close(
            prefix_output,
            full_output[:history_len],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_batched_variable_target_counts_match_full_forward(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_lens = [7, 11]
        history_lens = [4, 6]
        num_targets = [3, 5]
        total_seq_len = sum(seq_lens)
        x = torch.randn(total_seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor(seq_lens, dtype=torch.int64),
            x_offsets=torch.tensor([0, 7, 18], dtype=torch.int64),
            max_seq_len=max(seq_lens),
            num_targets=torch.tensor(num_targets, dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((2, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 7, 18], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 7, 18], dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
            _seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0, 0], dtype=torch.int32),
            num_reqs=2,
            num_actual_tokens=total_seq_len,
            max_query_len=max(seq_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
            slot_mapping=torch.tensor(
                list(range(seq_lens[0])) + [16 + i for i in range(seq_lens[1])],
                dtype=torch.int64,
            ),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor(num_targets, dtype=torch.int32),
            history_lengths=torch.tensor(history_lens, dtype=torch.int32),
            contextual_seq_len=0,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_x = torch.cat(
            [
                x[0:history_lens[0]],
                x[seq_lens[0]:seq_lens[0] + history_lens[1]],
            ],
            dim=0,
        )
        delta_x = torch.cat(
            [
                x[history_lens[0]:seq_lens[0]],
                x[seq_lens[0] + history_lens[1]:total_seq_len],
            ],
            dim=0,
        )
        prefix_output = native_layer(
            x=prefix_x,
            x_lengths=torch.tensor(history_lens, dtype=torch.int64),
            x_offsets=torch.tensor([0, 4, 10], dtype=torch.int64),
            max_seq_len=max(history_lens),
            num_targets=torch.tensor([0, 0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )
        delta_output = native_layer.cached_forward(
            delta_x=delta_x,
            num_targets=torch.tensor(num_targets, dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="delta",
        )

        combined = torch.cat(
            [
                prefix_output[0:history_lens[0]],
                delta_output[0:num_targets[0]],
                prefix_output[history_lens[0]:sum(history_lens)],
                delta_output[num_targets[0]:sum(num_targets)],
            ],
            dim=0,
        )

        torch.testing.assert_close(
            combined,
            full_output,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_contextual_prefix_metadata_matches_full_forward_semantics(self) -> None:
        full_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        full_layer.eval()
        native_layer = STULayer(
            STULayerConfig(
                embedding_dim=6,
                num_heads=2,
                hidden_dim=3,
                attention_dim=2,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                contextual_seq_len=0,
            )
        )
        native_layer.load_state_dict(full_layer.state_dict())
        native_layer.eval()

        seq_len = 9
        contextual_seq_len = 2
        history_len = 4
        num_targets = 3
        prefix_len = contextual_seq_len + history_len
        x = torch.randn(seq_len, 6, dtype=torch.float32)

        full_output = full_layer(
            x=x,
            x_lengths=torch.tensor([seq_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, seq_len], dtype=torch.int64),
            max_seq_len=seq_len,
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
        )

        cache_layer = HSTUKVCacheAttentionLayer(
            layer_name="hstu_kv_cache_layers.0",
            num_heads=2,
            head_size=2,
            head_size_v=3,
            dtype=torch.float32,
            vllm_config=None,
            enable_kv_cache=True,
        )
        runtime_kv_cache = torch.zeros((1, 16, 2, 5), dtype=torch.float32)
        cache_layer.get_runtime_kv_cache = (
            lambda kv_cache=None, _kv=runtime_kv_cache: _kv if kv_cache is None else kv_cache
        )

        common = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, seq_len], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([0], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.arange(seq_len, dtype=torch.int64),
            causal=True,
        )
        metadata = HSTUAttentionMetadata(
            common_attn_metadata=common,
            num_targets=torch.tensor([num_targets], dtype=torch.int32),
            history_lengths=torch.tensor([history_len], dtype=torch.int32),
            contextual_seq_len=contextual_seq_len,
            layer_names=["hstu_kv_cache_layers.0"],
            cache_layers=[cache_layer],
        )

        prefix_output = native_layer(
            x=x[:prefix_len],
            x_lengths=torch.tensor([prefix_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, prefix_len], dtype=torch.int64),
            max_seq_len=prefix_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )
        delta_output = native_layer.cached_forward(
            delta_x=x[prefix_len:],
            num_targets=torch.tensor([num_targets], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="delta",
        )

        torch.testing.assert_close(
            torch.cat([prefix_output, delta_output], dim=0),
            full_output,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_native_prefix_projection_uses_full_sequence_shape(self) -> None:
        seq_len = 9
        history_len = 6
        num_targets = 3
        native_layer, cache_layer, metadata, x = self._build_native_kv_test_fixture(
            seq_len=seq_len,
            history_len=history_len,
            num_targets=num_targets,
        )

        captured_rows: list[int] = []
        original_uqvk = stu_module.hstu_compute_uqvk

        def _capture_uqvk(*args, **kwargs):
            captured_rows.append(int(kwargs["x"].shape[0]))
            return original_uqvk(*args, **kwargs)

        with mock.patch.object(stu_module, "hstu_compute_uqvk", side_effect=_capture_uqvk):
            native_layer(
                x=x[:history_len],
                x_lengths=torch.tensor([history_len], dtype=torch.int64),
                x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
                max_seq_len=history_len,
                num_targets=torch.tensor([0], dtype=torch.int64),
                hstu_attn_metadata=metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode="prefix",
            )

        self.assertEqual(captured_rows, [seq_len])

    def test_native_delta_projection_uses_full_sequence_shape(self) -> None:
        seq_len = 9
        history_len = 6
        num_targets = 3
        native_layer, cache_layer, metadata, x = self._build_native_kv_test_fixture(
            seq_len=seq_len,
            history_len=history_len,
            num_targets=num_targets,
        )

        native_layer(
            x=x[:history_len],
            x_lengths=torch.tensor([history_len], dtype=torch.int64),
            x_offsets=torch.tensor([0, history_len], dtype=torch.int64),
            max_seq_len=history_len,
            num_targets=torch.tensor([0], dtype=torch.int64),
            hstu_attn_metadata=metadata,
            hstu_cache_layer=cache_layer,
            hstu_attention_mode="prefix",
        )

        captured_rows: list[int] = []
        original_uqvk = stu_module.hstu_compute_uqvk

        def _capture_uqvk(*args, **kwargs):
            captured_rows.append(int(kwargs["x"].shape[0]))
            return original_uqvk(*args, **kwargs)

        with mock.patch.object(stu_module, "hstu_compute_uqvk", side_effect=_capture_uqvk):
            native_layer.cached_forward(
                delta_x=x[history_len:],
                num_targets=torch.tensor([num_targets], dtype=torch.int64),
                hstu_attn_metadata=metadata,
                hstu_cache_layer=cache_layer,
                hstu_attention_mode="delta",
            )

        self.assertEqual(captured_rows, [seq_len])

    def test_prefix_hit_coverage_falls_back_without_warning(self) -> None:
        model = self._make_hstu_model_for_metadata_test()
        metadata = self._make_attention_metadata_for_coverage_test(
            slot_count=1,
            computed_tokens=16,
        )
        fake_forward_context = SimpleNamespace(attn_metadata={"layer0": metadata})

        with mock.patch(
            "vllm.forward_context.is_forward_context_available",
            return_value=True,
        ), mock.patch(
            "vllm.forward_context.get_forward_context",
            return_value=fake_forward_context,
        ), mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.debug_once"
        ) as debug_once, mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.warning_once"
        ) as warning_once:
            result = model._get_hstu_attention_metadata()

        self.assertIsNone(result)
        debug_once.assert_called_once()
        warning_once.assert_not_called()

    def test_incomplete_coverage_keeps_warning(self) -> None:
        model = self._make_hstu_model_for_metadata_test()
        metadata = self._make_attention_metadata_for_coverage_test(
            slot_count=1,
            computed_tokens=0,
        )
        fake_forward_context = SimpleNamespace(attn_metadata={"layer0": metadata})

        with mock.patch(
            "vllm.forward_context.is_forward_context_available",
            return_value=True,
        ), mock.patch(
            "vllm.forward_context.get_forward_context",
            return_value=fake_forward_context,
        ), mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.debug_once"
        ) as debug_once, mock.patch(
            "vllm_gr.model_executor.models.hstu_vllm.logger.warning_once"
        ) as warning_once:
            result = model._get_hstu_attention_metadata()

        self.assertIsNone(result)
        debug_once.assert_not_called()
        warning_once.assert_called_once()
