#!/usr/bin/env python3
# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""
Convert DLRM_v3 training checkpoints to HuggingFace-compatible format.

Meta's generative-recommenders saves training checkpoints in a split format:
    {path}/{batch_idx}/
    ├── sparse/            # torch.distributed.checkpoint (embeddings)
    └── non_sparse.ckpt    # torch.save dict with dense_dict, optimizer, etc.

This script converts them into vLLM-compatible flat format:
    {output_dir}/
    ├── config.json        # HSTUConfig (must be provided or generated)
    └── model.safetensors  # flat state_dict with "hstu." prefix

Reuses the upstream checkpoint utilities ported in
vllm_gr.models.dlrm_v3.checkpoint for maximum code fidelity.

Dense weights
=============
Loaded from non_sparse.ckpt via torch.load(). The checkpoint contains a
``dense_dict`` with all non-embedding parameters.  Sparse keys are
filtered out with ``is_sparse_key()``, the ``hstu.`` prefix is added,
and tensors are cast to the target dtype (bfloat16 by default).

Sparse (embedding) weights
===========================
Saved as ``ShardedTensor`` via PyTorch Distributed Checkpoint (DCP) in the
``sparse/`` directory.  During training, embedding tables may be sharded
across N GPUs.  To load them in a single-process conversion script:

1. ``torch.distributed`` is initialized with the ``gloo`` backend
   (world_size=1, rank=0).
2. A ``DlrmHSTU`` model is created with ``EmbeddingCollection`` on CPU.
3. Dense weights are loaded into the model first (required by
   ``load_dense_state_dict``'s assertion).
4. ``load_sparse_checkpoint(model, path)`` calls DCP, which automatically
   reshards the N-rank checkpoint into the single-rank model.
5. Embedding tensors are extracted from the model's state_dict.

Alternatively, run via ``torchrun`` to let PyTorch handle distributed init::

    torchrun --nproc_per_node=1 scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json

Usage:
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --dataset movielens-1m

    # With existing config.json:
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json

    # Dense-only (skip embeddings, matching upstream inference pattern):
    python scripts/convert_gr_checkpoint.py \\
        --input-path /path/to/checkpoint/{batch_idx} \\
        --output-dir /path/to/hf_model \\
        --config-json /path/to/config.json \\
        --dense-only
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vllm_gr.models.dlrm_v3.checkpoint import (
    is_sparse_key,
    load_dense_state_dict,
    load_sparse_checkpoint,
)

logger = logging.getLogger(__name__)


def _load_supported_dataset_choices() -> tuple[str, ...]:
    """Load supported dataset names without importing the heavy config package."""
    module_name = "vllm_gr_dataset_configs_for_conversion"
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vllm_gr",
        "config",
        "dataset_configs.py",
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load dataset configs from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.SUPPORTED_DATASETS)


DATASET_CHOICES = _load_supported_dataset_choices()


def _resolve_dtype_name(dtype_name: str, *, default: torch.dtype) -> torch.dtype:
    normalized = str(dtype_name).lower()
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(normalized, default)


def _build_embedding_table_dtype_map(config: Any) -> Dict[str, torch.dtype]:
    default_dtype = _resolve_dtype_name(
        getattr(config, "embedding_table_data_type", "fp16"),
        default=torch.float16,
    )
    table_dtypes: Dict[str, torch.dtype] = {}

    for table_cfg in getattr(config, "embedding_tables", []) or []:
        table_name = str(table_cfg["name"])
        table_dtypes[table_name] = _resolve_dtype_name(
            table_cfg.get("data_type", default_dtype),
            default=default_dtype,
        )

    return table_dtypes


def _resolve_sparse_tensor_target_dtype(
    config: Any,
    tensor_name: str,
    tensor: torch.Tensor,
) -> torch.dtype:
    if not torch.is_floating_point(tensor):
        return tensor.dtype

    table_dtypes = _build_embedding_table_dtype_map(config)
    for table_name, table_dtype in table_dtypes.items():
        if f".{table_name}." in tensor_name:
            return table_dtype
    return tensor.dtype


def _init_single_process_distributed() -> None:
    """Initialize torch.distributed for single-process DCP loading.

    Uses gloo backend (CPU-only, no GPU required). Safe to call multiple
    times -- returns immediately if already initialized.
    """
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="gloo")
    logger.info("Initialized single-process torch.distributed (gloo)")


def _destroy_distributed() -> None:
    """Destroy torch.distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed torch.distributed process group")


def load_dense_weights(
    checkpoint_path: str,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load dense (non-embedding) weights from non_sparse.ckpt.

    Reads the checkpoint and filters out sparse keys using the upstream
    is_sparse_key function from checkpoint.py.

    Args:
        checkpoint_path: Path to checkpoint directory containing non_sparse.ckpt.
        device: Device to load tensors onto.

    Returns:
        Dictionary of dense parameter tensors.
    """
    non_sparse_ckpt = os.path.join(checkpoint_path, "non_sparse.ckpt")
    if not os.path.exists(non_sparse_ckpt):
        raise FileNotFoundError(f"non_sparse.ckpt not found at {non_sparse_ckpt}")

    logger.info("Loading dense weights from %s", non_sparse_ckpt)
    state = torch.load(non_sparse_ckpt, map_location=device, weights_only=False)
    dense_dict = state["dense_dict"]

    filtered = {}
    for k, v in dense_dict.items():
        if is_sparse_key(k, v):
            logger.debug("Skipping sparse key in dense_dict: %s", k)
            continue
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        filtered[k] = v

    logger.info("Loaded %d dense tensors", len(filtered))
    return filtered


def load_sparse_weights(
    checkpoint_path: str,
    config_dir: str,
    dense_weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Load sparse (embedding) weights via DCP and the upstream load_sparse_checkpoint.

    Creates a DlrmHSTU model, loads dense weights into it first (required by
    the upstream ``load_dense_state_dict`` assertion), then calls
    ``load_sparse_checkpoint()`` to load DCP shards into the model. Extracts
    and returns the embedding tensors.

    Requires ``torch.distributed`` to be initialized (call
    ``_init_single_process_distributed()`` first).

    Args:
        checkpoint_path: Path to checkpoint directory containing sparse/.
        config_dir: Path to directory with config.json (for model creation).
        dense_weights: Dense weights dict (keys without hstu. prefix) to
            pre-load into the model before sparse loading.

    Returns:
        Dictionary of sparse parameter tensors (keys without hstu. prefix).
    """
    sparse_path = os.path.join(checkpoint_path, "sparse")
    if not os.path.isdir(sparse_path):
        logger.info("No sparse/ directory found at %s, skipping", sparse_path)
        return {}

    from vllm_gr.config.hstu_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    config = HSTUConfig.from_pretrained(config_dir)
    model = HSTUForRecommendation(config=config)
    hstu = model.hstu

    # Load dense weights into the model first -- load_sparse_checkpoint's
    # upstream SparseState depends on the model having consistent state.
    load_dense_state_dict(hstu, dense_weights)
    logger.info("Pre-loaded dense weights into model for sparse loading")

    sparse_keys_before = {
        k for k, v in hstu.state_dict().items() if is_sparse_key(k, v)
    }
    if not sparse_keys_before:
        raise RuntimeError(
            "Sparse checkpoint/config mismatch: checkpoint contains sparse weights, "
            "but the instantiated model exposes no sparse parameters. "
            "Check that config.json matches the checkpoint layout."
        )

    logger.info(
        "Loading sparse weights (%d embedding tensors) via DCP from %s",
        len(sparse_keys_before),
        sparse_path,
    )
    load_sparse_checkpoint(model=hstu, path=checkpoint_path)

    result = {}
    for k, v in hstu.state_dict().items():
        if k in sparse_keys_before:
            result[k] = v.detach().cpu()
    logger.info("Loaded %d sparse tensors", len(result))
    return result


def generate_config_from_dataset(
    dataset: str,
    output_dir: str,
) -> None:
    """Generate a config.json for the given dataset using vllm-GR dataset configs."""
    from vllm_gr.config.hstu_config import HSTUConfig
    from vllm_gr.config.dataset_configs import (
        get_hstu_config_overrides,
        get_embedding_table_overrides,
    )

    overrides = get_hstu_config_overrides(dataset)
    tables = get_embedding_table_overrides(dataset)
    if tables:
        overrides["embedding_tables"] = tables

    config = HSTUConfig(**overrides)
    config.save_pretrained(output_dir)
    logger.info("Generated config.json for dataset=%s at %s", dataset, output_dir)


def convert(
    input_path: str,
    output_dir: str,
    config_json: Optional[str] = None,
    dataset: Optional[str] = None,
    dense_only: bool = False,
    output_format: str = "safetensors",
    target_dtype: str = "bfloat16",
) -> str:
    """Convert DLRM_v3 checkpoint to HuggingFace-compatible format.

    Args:
        input_path: Path to checkpoint dir (containing non_sparse.ckpt).
        output_dir: Output directory for converted model.
        config_json: Path to existing config.json to copy.
        dataset: Dataset name to generate config from.
        dense_only: If True, skip sparse (embedding) weights.
        output_format: "safetensors" or "pt".
        target_dtype: Target dtype for dense tensors (default: "bfloat16").
            Sparse embedding tensors follow config.json table dtype settings.

    Returns:
        Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    torch_dtype = _resolve_dtype_name(target_dtype, default=torch.bfloat16)
    hf_config = None

    # Handle config.json
    if config_json:
        shutil.copy2(config_json, os.path.join(output_dir, "config.json"))
        logger.info("Copied config.json from %s", config_json)
    elif dataset:
        generate_config_from_dataset(dataset, output_dir)

    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        from vllm_gr.config.hstu_config import HSTUConfig
        hf_config = HSTUConfig.from_pretrained(output_dir)

    # Load dense weights
    dense_dict = load_dense_weights(input_path)

    # Build converted dict with "hstu." prefix and dtype casting
    converted: Dict[str, torch.Tensor] = {}
    for k, v in dense_dict.items():
        converted["hstu." + k] = v.to(torch_dtype)

    # Load sparse weights if requested
    has_sparse_dir = os.path.isdir(os.path.join(input_path, "sparse"))
    include_sparse = has_sparse_dir and not dense_only

    if include_sparse:
        if not os.path.exists(config_path):
            raise RuntimeError(
                "Sparse checkpoint conversion requires config.json when sparse/ "
                "weights are present. Provide --config-json, --dataset, or "
                "use --dense-only."
            )
        else:
            _init_single_process_distributed()
            try:
                sparse_dict = load_sparse_weights(
                    input_path, output_dir, dense_dict,
                )
                for k, v in sparse_dict.items():
                    sparse_target_dtype = (
                        _resolve_sparse_tensor_target_dtype(hf_config, k, v)
                        if hf_config is not None
                        else v.dtype
                    )
                    converted["hstu." + k] = v.to(sparse_target_dtype)
            except Exception as e:
                logger.error(
                    "Sparse loading failed: %s\n"
                    "Ensure torchrec is installed. You can also try:\n"
                    "  torchrun --nproc_per_node=1 scripts/convert_gr_checkpoint.py ...\n"
                    "Or use --dense-only to skip embeddings.",
                    e,
                )
                raise
            finally:
                _destroy_distributed()
    elif has_sparse_dir and dense_only:
        logger.info(
            "Sparse directory found but --dense-only specified, skipping embeddings"
        )

    # Save
    if output_format == "safetensors":
        try:
            from safetensors.torch import save_file
            out_file = os.path.join(output_dir, "model.safetensors")
            save_file(converted, out_file)
        except ImportError:
            raise RuntimeError(
                "Output format 'safetensors' requested, but safetensors is not "
                "installed. Install `safetensors` or use `--format pt`."
            )
    else:
        out_file = os.path.join(output_dir, "model.pt")
        torch.save(converted, out_file)

    total_params = sum(v.numel() for v in converted.values())
    sparse_count = sum(1 for k in converted if "embedding_collection" in k)
    dense_count = len(converted) - sparse_count
    logger.info(
        "Converted %d tensors (%d parameters) -> %s",
        len(converted), total_params, out_file,
    )
    sparse_dtype_summary = (
        "per-table config"
        if sparse_count and hf_config is not None
        else ("original tensor dtypes" if sparse_count else "n/a")
    )
    logger.info(
        "  Dense: %d tensors (target dtype=%s), Sparse: %d tensors "
        "(target dtype=%s)",
        dense_count, torch_dtype, sparse_count, sparse_dtype_summary,
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DLRM_v3 training checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-path", required=True,
        help="Path to DLRM_v3 checkpoint directory (containing non_sparse.ckpt)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for converted HuggingFace model",
    )
    parser.add_argument(
        "--config-json", default=None,
        help="Path to existing config.json to copy into output directory",
    )
    parser.add_argument(
        "--dataset", default=None,
        choices=DATASET_CHOICES,
        help="Dataset name to auto-generate config.json",
    )
    parser.add_argument(
        "--dense-only", action="store_true", default=False,
        help=(
            "Only convert dense weights (skip sparse embeddings). "
            "Matches upstream inference pattern where embeddings come "
            "from an external feature store."
        ),
    )
    parser.add_argument(
        "--format", choices=["safetensors", "pt"], default="safetensors",
        help="Output weight format (default: safetensors)",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Target dtype for all tensors (default: bfloat16)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    convert(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config_json=args.config_json,
        dataset=args.dataset,
        dense_only=args.dense_only,
        output_format=args.format,
        target_dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
