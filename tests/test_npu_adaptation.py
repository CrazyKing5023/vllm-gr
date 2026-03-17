# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("vllm")
pytest.importorskip("torchrec")

from vllm_gr.models.dlrm_v3.common import HammerKernel
from vllm_gr.ops.hstu_attention import hstu_mha, hstu_mha_npu
from vllm_gr.utils import (
    DeviceAdapter,
    DeviceType,
    ReliableOps,
    get_device_backend,
    is_npu_device,
    move_hstu_sparse_outputs_to_device,
    resolve_hstu_dense_device,
    resolve_hstu_embedding_device,
)
from vllm_gr.worker.gr_runner_mixin import GRRunnerMixin


class _DummyRunner(GRRunnerMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        self.input_batch = SimpleNamespace(req_ids=["r1"])
        self._gr_features_cache = {}


def test_device_adapter_reports_a_supported_backend():
    adapter = DeviceAdapter()
    assert adapter.device_backend in {
        DeviceType.CPU,
        DeviceType.CUDA,
        DeviceType.NPU,
        DeviceType.UNKNOWN,
    }


def test_reliable_ops_cumsum_falls_back_on_cpu():
    lengths = torch.tensor([1, 2, 3], dtype=torch.int64)
    result = ReliableOps.asynchronous_complete_cumsum(lengths)
    expected = torch.tensor([0, 1, 3, 6], dtype=torch.int64)
    torch.testing.assert_close(result, expected)


def test_common_exposes_npu_backend_helpers():
    cpu_device = torch.device("cpu")
    assert get_device_backend(cpu_device) == DeviceType.CPU
    assert not is_npu_device(cpu_device)


def test_hstu_runtime_helpers_prefer_explicit_devices():
    module = torch.nn.Linear(4, 4)
    cpu_device = torch.device("cpu")

    assert resolve_hstu_dense_device(module, explicit_device=cpu_device) == cpu_device
    assert resolve_hstu_embedding_device() == cpu_device


def test_hstu_runtime_moves_sparse_outputs_to_target_device():
    from vllm_gr.models.dlrm_v3.modules.dlrm_hstu import SequenceEmbedding

    seq_embeddings = {
        "item_id": SequenceEmbedding(
            lengths=torch.tensor([2], dtype=torch.int64),
            embedding=torch.randn(2, 4),
        ),
    }
    payload_features = {
        "uih_offsets": torch.tensor([0, 2], dtype=torch.int64),
    }

    (
        moved_embeddings,
        moved_payloads,
        moved_lengths,
        moved_candidates,
    ) = move_hstu_sparse_outputs_to_device(
        seq_embeddings=seq_embeddings,
        payload_features=payload_features,
        uih_seq_lengths=torch.tensor([2], dtype=torch.int64),
        num_candidates=torch.tensor([1], dtype=torch.int64),
        device=torch.device("cpu"),
    )

    assert moved_embeddings["item_id"].embedding.device.type == "cpu"
    assert moved_payloads["uih_offsets"].device.type == "cpu"
    assert moved_lengths.device.type == "cpu"
    assert moved_candidates.device.type == "cpu"


def test_hstu_attention_exposes_npu_entrypoint():
    assert HammerKernel.NPU == HammerKernel.NPU
    assert callable(hstu_mha)
    assert callable(hstu_mha_npu)


def test_gr_runner_builds_cpu_kjts_from_current_payload_shape():
    runner = _DummyRunner()
    payload = SimpleNamespace(
        uih_keys=["item_id"],
        uih_values_per_key=[[1, 2]],
        uih_lengths_per_key=[2],
        cand_keys=["item_candidate_id"],
        cand_values_per_key=[[11, 12]],
        cand_lengths_per_key=[2],
        ctx_keys=["country"],
        ctx_values_per_key=[[86]],
    )

    decoded = runner._decode_gr_features(payload)
    runner._gr_features_cache["r1"] = decoded

    model_kwargs = runner._build_gr_model_kwargs()

    assert set(model_kwargs) == {"uih_features", "candidates_features"}
    assert model_kwargs["uih_features"].keys() == ["item_id", "country"]
    assert model_kwargs["candidates_features"].keys() == ["item_candidate_id"]
    assert model_kwargs["uih_features"].values().device.type == "cpu"
    assert model_kwargs["candidates_features"].values().device.type == "cpu"


def test_gr_runner_rejects_mixed_batch_schema_mismatch():
    runner = _DummyRunner()
    runner.input_batch = SimpleNamespace(req_ids=["r1", "r2"])

    payload1 = SimpleNamespace(
        uih_keys=["item_id"],
        uih_values_per_key=[[1, 2]],
        uih_lengths_per_key=[2],
        cand_keys=["item_candidate_id"],
        cand_values_per_key=[[11, 12]],
        cand_lengths_per_key=[2],
        ctx_keys=["country"],
        ctx_values_per_key=[[86]],
    )
    payload2 = SimpleNamespace(
        uih_keys=["item_id"],
        uih_values_per_key=[[3, 4]],
        uih_lengths_per_key=[2],
        cand_keys=["item_candidate_id"],
        cand_values_per_key=[[13, 14]],
        cand_lengths_per_key=[2],
        ctx_keys=["country", "gender"],
        ctx_values_per_key=[[86], [1]],
    )

    runner._gr_features_cache["r1"] = runner._decode_gr_features(payload1)
    runner._gr_features_cache["r2"] = runner._decode_gr_features(payload2)

    with pytest.raises(ValueError, match="Mixed GR batch schema mismatch"):
        runner._build_gr_model_kwargs()


def test_arg_utils_selects_npu_worker_when_platform_is_ascend():
    pytest.importorskip("vllm")
    from vllm_gr.engine.arg_utils import GREngineArgs

    with mock.patch.object(GREngineArgs, "_is_ascend_platform", return_value=True):
        args = GREngineArgs(model="")

    assert args.worker_cls == "vllm_gr.worker.gr_npu_worker.GRNPUWorker"
