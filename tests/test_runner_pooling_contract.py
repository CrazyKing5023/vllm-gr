# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Unit tests for GR runner pooling compatibility."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import logging
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class _FakeModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    kv_connector_output: object | None = None
    pooler_output: list[torch.Tensor | None] | None = None


class _FakePoolingMetadata:
    def __init__(self, prompt_lens):
        self.prompt_lens = torch.tensor(prompt_lens, dtype=torch.int64)
        self.cursor_device = None

    def build_pooling_cursor(self, _num_scheduled_tokens_np, _seq_lens_cpu, device):
        self.cursor_device = device


class _FakeInputBatch:
    def __init__(self):
        self.num_reqs = 2
        self.pooling_params = [object(), object()]
        self.req_ids = ["req-1", "req-2"]
        self.req_id_to_index = {"req-1": 0, "req-2": 1}
        self._pooling_metadata = _FakePoolingMetadata([2, 3])

    def get_pooling_metadata(self):
        return self._pooling_metadata


class TestRunnerPoolingContract(unittest.TestCase):

    def test_gr_runner_pool_accepts_per_request_hidden_state_list(self):
        fake_vllm_logger = types.ModuleType("vllm.logger")
        fake_vllm_logger.init_logger = logging.getLogger
        fake_vllm_sequence = types.ModuleType("vllm.sequence")
        fake_vllm_sequence.IntermediateTensors = dict
        fake_vllm_outputs = types.ModuleType("vllm.v1.outputs")
        fake_vllm_outputs.ModelRunnerOutput = _FakeModelRunnerOutput
        fake_torchrec = types.ModuleType("torchrec")
        fake_torchrec_sparse = types.ModuleType("torchrec.sparse")
        fake_torchrec_jt = types.ModuleType("torchrec.sparse.jagged_tensor")
        fake_torchrec_jt.KeyedJaggedTensor = object

        with mock.patch.dict(
            sys.modules,
            {
                "vllm.logger": fake_vllm_logger,
                "vllm.sequence": fake_vllm_sequence,
                "vllm.v1.outputs": fake_vllm_outputs,
                "torchrec": fake_torchrec,
                "torchrec.sparse": fake_torchrec_sparse,
                "torchrec.sparse.jagged_tensor": fake_torchrec_jt,
            },
        ):
            module_path = ROOT / "vllm_gr" / "worker" / "gr_runner_mixin.py"
            spec = importlib.util.spec_from_file_location(
                "gr_runner_mixin_under_test",
                module_path,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            GRRunnerMixin = module.GRRunnerMixin

            class _FakeRunner(GRRunnerMixin):
                def __init__(self):
                    self.input_batch = _FakeInputBatch()
                    self.seq_lens = SimpleNamespace(
                        cpu=torch.tensor([2, 3], dtype=torch.int64),
                    )
                    self.sync_called = False

                    class _FakeModel:
                        @staticmethod
                        def pooler(hidden_states, pooling_metadata):
                            return hidden_states

                    self.model = _FakeModel()

                def _sync_device(self):
                    self.sync_called = True

            runner = _FakeRunner()
            outputs = [
                torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
            ]

            result = runner._pool(
                hidden_states=outputs,
                num_scheduled_tokens=5,
                num_scheduled_tokens_np=np.array([2, 3], dtype=np.int32),
                kv_connector_output=None,
            )

            self.assertEqual(result.req_ids, ["req-1", "req-2"])
            self.assertEqual(len(result.pooler_output), 2)
            self.assertEqual(result.pooler_output[0].device.type, "cpu")
            self.assertEqual(result.pooler_output[1].device.type, "cpu")
            self.assertEqual(
                runner.input_batch.get_pooling_metadata().cursor_device.type,
                outputs[0].device.type,
            )
            self.assertTrue(runner.sync_called)


if __name__ == "__main__":
    unittest.main()
