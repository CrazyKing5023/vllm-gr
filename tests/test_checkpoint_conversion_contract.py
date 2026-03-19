# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "convert_gr_checkpoint.py"
DATASET_CONFIGS_PATH = ROOT / "vllm_gr" / "config" / "dataset_configs.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestCheckpointConversionContract(unittest.TestCase):

    def test_dataset_choices_follow_supported_dataset_registry(self):
        fake_checkpoint_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.checkpoint"
        )
        fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
        fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
        fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_choices_test_module",
                MODULE_PATH,
            )

        dataset_module = _load_module(
            "checkpoint_conversion_dataset_configs_test_module",
            DATASET_CONFIGS_PATH,
        )

        self.assertEqual(
            module.DATASET_CHOICES,
            tuple(dataset_module.SUPPORTED_DATASETS),
        )

    def test_sparse_embedding_dtype_follows_table_config(self):
        fake_checkpoint_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.checkpoint"
        )
        fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
        fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
        fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_contract_test_module",
                MODULE_PATH,
            )

        config = SimpleNamespace(
            embedding_table_data_type="fp16",
            embedding_tables=[
                {"name": "item_id", "data_type": "bf16"},
                {"name": "user_id"},
            ],
        )

        item_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.item_id.weight",
            torch.ones(2, dtype=torch.float32),
        )
        user_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.user_id.weight",
            torch.ones(2, dtype=torch.float32),
        )
        int_dtype = module._resolve_sparse_tensor_target_dtype(
            config,
            "_embedding_collection.embeddings.user_id.offsets",
            torch.ones(2, dtype=torch.int64),
        )

        self.assertEqual(item_dtype, torch.bfloat16)
        self.assertEqual(user_dtype, torch.float16)
        self.assertEqual(int_dtype, torch.int64)

    def test_convert_fails_when_sparse_weights_exist_without_config(self):
        fake_checkpoint_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.checkpoint"
        )
        fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
        fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
        fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_sparse_fail_test_module",
                MODULE_PATH,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            (input_dir / "sparse").mkdir(parents=True)

            with mock.patch.object(module, "load_dense_weights", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "requires config.json"):
                    module.convert(
                        input_path=str(input_dir),
                        output_dir=str(output_dir),
                        dense_only=False,
                        output_format="pt",
                    )

    def test_load_sparse_weights_fails_when_model_exposes_no_sparse_params(self):
        fake_checkpoint_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.checkpoint"
        )
        fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
        fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
        fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_sparse_mismatch_test_module",
                MODULE_PATH,
            )

        fake_config_module = types.ModuleType("vllm_gr.config.hstu_config")
        fake_model_module = types.ModuleType("vllm_gr.model_executor.models.hstu_vllm")

        class _FakeHSTUConfig:
            @classmethod
            def from_pretrained(cls, _config_dir):
                return object()

        class _FakeInnerModel:
            @staticmethod
            def state_dict():
                return {}

        class _FakeHSTUForRecommendation:
            def __init__(self, config):
                self.hstu = _FakeInnerModel()

        fake_config_module.HSTUConfig = _FakeHSTUConfig
        fake_model_module.HSTUForRecommendation = _FakeHSTUForRecommendation

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "sparse").mkdir()

            with mock.patch.dict(
                sys.modules,
                {
                    "vllm_gr.config.hstu_config": fake_config_module,
                    "vllm_gr.model_executor.models.hstu_vllm": fake_model_module,
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Sparse checkpoint/config mismatch"):
                    module.load_sparse_weights(
                        checkpoint_path=str(checkpoint_dir),
                        config_dir=str(root),
                        dense_weights={},
                    )

    def test_convert_fails_when_safetensors_requested_but_missing(self):
        fake_checkpoint_module = types.ModuleType(
            "vllm_gr.models.dlrm_v3.checkpoint"
        )
        fake_checkpoint_module.is_sparse_key = lambda *args, **kwargs: False
        fake_checkpoint_module.load_dense_state_dict = lambda *args, **kwargs: None
        fake_checkpoint_module.load_sparse_checkpoint = lambda *args, **kwargs: None

        with mock.patch.dict(
            sys.modules,
            {"vllm_gr.models.dlrm_v3.checkpoint": fake_checkpoint_module},
        ):
            module = _load_module(
                "checkpoint_conversion_safetensors_fail_test_module",
                MODULE_PATH,
            )

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "safetensors.torch" or (
                name == "safetensors" and fromlist and "torch" in fromlist
            ):
                raise ImportError("missing safetensors")
            return real_import(name, globals, locals, fromlist, level)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            with mock.patch.object(module, "load_dense_weights", return_value={}):
                with mock.patch("builtins.__import__", side_effect=_fake_import):
                    with self.assertRaisesRegex(RuntimeError, "safetensors.*not installed"):
                        module.convert(
                            input_path=str(input_dir),
                            output_dir=str(output_dir),
                            dense_only=True,
                            output_format="safetensors",
                        )


if __name__ == "__main__":
    unittest.main()
