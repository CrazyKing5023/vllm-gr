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
filtered out with ``is_sparse_key()``. Floating-point dense tensors are
converted to ``bfloat16`` while integer tensors (for example ``int64``
buffers) are preserved as-is. The ``hstu.`` prefix is added before
saving. Source dense floating-point tensors whose dtype is not already
``bfloat16`` are reported as warnings.

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

Note: DCP loads tensors into pre-allocated model tensors. If ckpt dtype and
model tensor dtype differ, values are cast to model dtype during loading.
This converter reports such dtype mismatches as warnings before loading.
The converter also logs dtype distributions for the original checkpoint
and for the final converted tensor set before save.

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
import collections
import json
import logging
import os
import shutil
import sys
from math import prod
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vllm_gr.models.dlrm_v3.checkpoint import (
    is_sparse_key,
    load_dense_state_dict,
    load_sparse_checkpoint,
)

logger = logging.getLogger(__name__)


def _read_sparse_metadata(checkpoint_path: str) -> Optional[Any]:
    """Read sparse DCP metadata once for mismatch checks."""
    sparse_path = os.path.join(checkpoint_path, "sparse")
    if not os.path.isdir(sparse_path):
        return None
    try:
        reader = torch.distributed.checkpoint.FileSystemReader(sparse_path)
        return reader.read_metadata()
    except Exception as exc:
        logger.warning(
            "Sparse dtype check skipped: failed to read DCP metadata from %s (%s)",
            sparse_path,
            exc,
        )
        return None


def _summarize_dtype_distribution_from_tensors(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, int]]:
    """Summarize dtype distribution from tensor dict."""
    stats: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"tensor_count": 0, "numel_count": 0}
    )
    for tensor in weights.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        dtype_key = str(tensor.dtype)
        stats[dtype_key]["tensor_count"] += 1
        stats[dtype_key]["numel_count"] += int(tensor.numel())
    return dict(stats)


def _summarize_dtype_distribution_from_sparse_metadata(
    metadata: Any,
) -> Dict[str, Dict[str, int]]:
    """Summarize dtype distribution from sparse DCP metadata."""
    stats: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: {"tensor_count": 0, "numel_count": 0}
    )
    metadata_items = getattr(metadata, "state_dict_metadata", None)
    if not isinstance(metadata_items, dict):
        return {}

    for meta_entry in metadata_items.values():
        props = getattr(meta_entry, "properties", None)
        dtype = getattr(props, "dtype", None)
        size = getattr(meta_entry, "size", None)
        if dtype is None or size is None:
            continue

        dtype_key = str(dtype)
        stats[dtype_key]["tensor_count"] += 1
        stats[dtype_key]["numel_count"] += int(prod(tuple(size)))

    return dict(stats)


def _log_dtype_distribution(scope: str, stats: Dict[str, Dict[str, int]]) -> None:
    """Log dtype distribution in a stable format."""
    total_tensors = sum(v["tensor_count"] for v in stats.values())
    total_numel = sum(v["numel_count"] for v in stats.values())
    logger.info(
        "%s dtype distribution: %d tensors, %d elements, %d dtypes",
        scope,
        total_tensors,
        total_numel,
        len(stats),
    )
    for dtype_key in sorted(stats.keys()):
        item = stats[dtype_key]
        logger.info(
            "  %s: dtype=%s tensors=%d numel=%d",
            scope,
            dtype_key,
            item["tensor_count"],
            item["numel_count"],
        )


def save_config_preserve_order(config: "HSTUConfig", output_dir: str) -> None:
    """Write config.json without key sorting to preserve dict insertion order."""
    config_path = os.path.join(output_dir, "config.json")
    # Match PretrainedConfig.save_pretrained default behavior (diff dict),
    # but keep key order stable by disabling JSON sort_keys.
    config_dict = config.to_diff_dict()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")


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


def _warn_dense_dtype_mismatch(
    model_state: Dict[str, Any],
    dense_weights: Dict[str, torch.Tensor],
    max_examples: int = 20,
) -> None:
    """Warn when dense ckpt dtype differs from model tensor dtype."""
    checked = 0
    mismatches = []

    for key, value in dense_weights.items():
        ckpt_tensor = value.data if isinstance(value, torch.nn.Parameter) else value
        model_tensor = model_state.get(key)
        if not isinstance(ckpt_tensor, torch.Tensor):
            continue
        if not isinstance(model_tensor, torch.Tensor):
            continue

        checked += 1
        if ckpt_tensor.dtype != model_tensor.dtype:
            mismatches.append((key, ckpt_tensor.dtype, model_tensor.dtype))

    if not mismatches:
        return

    logger.warning(
        "Dense dtype mismatch detected before load_dense_state_dict: "
        "checked=%d, mismatch=%d. Loading still follows model tensor dtype.",
        checked,
        len(mismatches),
    )
    for idx, (key, ckpt_dtype, model_dtype) in enumerate(mismatches[:max_examples], 1):
        logger.warning(
            "Dense dtype mismatch %d/%d: key=%s, ckpt_dtype=%s, model_dtype=%s",
            idx,
            len(mismatches),
            key,
            ckpt_dtype,
            model_dtype,
        )


def _cast_dense_weights_to_bf16(
    dense_weights: Dict[str, torch.Tensor],
    max_examples: int = 20,
) -> Dict[str, torch.Tensor]:
    """Cast dense floating-point tensors to bf16 and preserve integer tensors."""
    target_dtype = torch.bfloat16
    non_bf16 = []
    casted: Dict[str, torch.Tensor] = {}

    for key, tensor in dense_weights.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        if not torch.is_floating_point(tensor):
            casted[key] = tensor
            continue

        if tensor.dtype != target_dtype:
            non_bf16.append((key, tensor.dtype))
        casted[key] = tensor.to(target_dtype)

    if non_bf16:
        logger.warning(
            "Dense floating-point source tensors will be converted to %s during "
            "conversion: checked=%d, non_bf16=%d.",
            target_dtype,
            len(dense_weights),
            len(non_bf16),
        )
        for idx, (key, src_dtype) in enumerate(non_bf16[:max_examples], 1):
            logger.warning(
                "Dense source dtype %d/%d: key=%s, src_dtype=%s, target_dtype=%s",
                idx,
                len(non_bf16),
                key,
                src_dtype,
                target_dtype,
            )

    return casted


def _warn_sparse_dtype_mismatch(
    model_state: Dict[str, Any],
    sparse_metadata: Any,
    max_examples: int = 20,
) -> None:
    """Warn when sparse ckpt dtype differs from model dtype.

    Mapping rule intentionally follows Kira's checkpoint layout:
    - exact key match, or
    - metadata key starts with "sparse_dict." and stripped key matches.
    """
    metadata_items = getattr(sparse_metadata, "state_dict_metadata", None)
    if not isinstance(metadata_items, dict):
        logger.warning(
            "Sparse dtype check skipped: invalid DCP metadata format (%s)",
            type(sparse_metadata).__name__,
        )
        return

    model_sparse = {
        key: tensor
        for key, tensor in model_state.items()
        if isinstance(tensor, torch.Tensor) and is_sparse_key(key, tensor)
    }
    if not model_sparse:
        return

    total_ckpt_sparse = 0
    checked = 0
    unresolved = 0
    mismatches = []
    for meta_key, meta_entry in metadata_items.items():
        meta_key = str(meta_key)
        props = getattr(meta_entry, "properties", None)
        ckpt_dtype = getattr(props, "dtype", None)
        if ckpt_dtype is None:
            continue

        total_ckpt_sparse += 1
        model_key: Optional[str] = None
        if meta_key in model_sparse:
            model_key = meta_key
        elif meta_key.startswith("sparse_dict."):
            stripped_key = meta_key[len("sparse_dict."):]
            if stripped_key in model_sparse:
                model_key = stripped_key

        if model_key is None:
            unresolved += 1
            continue

        model_dtype = model_sparse[model_key].dtype
        checked += 1
        if ckpt_dtype != model_dtype:
            mismatches.append((str(meta_key), model_key, ckpt_dtype, model_dtype))

    if total_ckpt_sparse > 0 and checked == 0:
        logger.warning(
            "Sparse dtype check skipped: no sparse metadata keys matched model keys "
            "under fixed mapping (exact or sparse_dict.*) "
            "(ckpt_sparse_keys=%d, unresolved=%d).",
            total_ckpt_sparse,
            unresolved,
        )
        return

    if not mismatches:
        return

    logger.warning(
        "Sparse dtype mismatch detected before load_sparse_checkpoint: "
        "ckpt_sparse_keys=%d, checked=%d, mismatch=%d, unresolved=%d. "
        "Loading still follows model tensor dtype.",
        total_ckpt_sparse,
        checked,
        len(mismatches),
        unresolved,
    )
    for idx, (meta_key, model_key, ckpt_dtype, model_dtype) in enumerate(
        mismatches[:max_examples], 1
    ):
        logger.warning(
            "Sparse dtype mismatch %d/%d: metadata_key=%s -> model_key=%s, "
            "ckpt_dtype=%s, model_dtype=%s",
            idx,
            len(mismatches),
            meta_key,
            model_key,
            ckpt_dtype,
            model_dtype,
        )


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
    sparse_metadata: Optional[Any] = None,
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
        sparse_metadata: Optional DCP metadata read once at higher level.

    Returns:
        Dictionary of sparse parameter tensors (keys without hstu. prefix).
    """
    sparse_path = os.path.join(checkpoint_path, "sparse")
    if not os.path.isdir(sparse_path):
        logger.warning("No sparse/ directory found at %s, skipping", sparse_path)
        return {}

    from vllm_gr.config.hstu_hf_config import HSTUConfig
    from vllm_gr.model_executor.models.hstu_vllm import HSTUForRecommendation

    config = HSTUConfig.from_pretrained(config_dir)
    model = HSTUForRecommendation(config=config)
    hstu = model.hstu
    hstu_state = hstu.state_dict()

    _warn_dense_dtype_mismatch(hstu_state, dense_weights)

    if sparse_metadata is not None:
        _warn_sparse_dtype_mismatch(hstu_state, sparse_metadata)
    else:
        logger.warning(
            "Sparse dtype check skipped: sparse metadata unavailable for %s",
            sparse_path,
        )

    # Load dense weights into the model first -- load_sparse_checkpoint's
    # upstream SparseState depends on the model having consistent state.
    load_dense_state_dict(hstu, dense_weights)
    logger.info("Pre-loaded dense weights into model for sparse loading")

    sparse_keys_before = {
        k for k, v in hstu.state_dict().items() if is_sparse_key(k, v)
    }
    if not sparse_keys_before:
        logger.info("Model has no sparse keys (is_dense=True?), skipping")
        return {}

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
    from vllm_gr.config.hstu_hf_config import HSTUConfig
    from vllm_gr.config.dataset_configs import (
        get_hstu_config_overrides,
        get_embedding_table_overrides,
    )

    overrides = get_hstu_config_overrides(dataset)
    tables = get_embedding_table_overrides(dataset)
    if tables:
        overrides["embedding_tables"] = tables

    config = HSTUConfig(**overrides)
    save_config_preserve_order(config, output_dir)
    logger.info("Generated config.json for dataset=%s at %s", dataset, output_dir)


def convert(
    input_path: str,
    output_dir: str,
    config_json: Optional[str] = None,
    dataset: Optional[str] = None,
    dense_only: bool = False,
    output_format: str = "safetensors",
) -> str:
    """Convert DLRM_v3 checkpoint to HuggingFace-compatible format.

    Args:
        input_path: Path to checkpoint dir (containing non_sparse.ckpt).
        output_dir: Output directory for converted model.
        config_json: Path to existing config.json to copy.
        dataset: Dataset name to generate config from.
        dense_only: If True, skip sparse (embedding) weights.
        output_format: "safetensors" or "pt".

    Returns:
        Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle config.json
    if config_json:
        shutil.copy2(config_json, os.path.join(output_dir, "config.json"))
        logger.info("Copied config.json from %s", config_json)
    elif dataset:
        generate_config_from_dataset(dataset, output_dir)
    else:
        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            logger.warning(
                "No config.json found/generated. Provide --config-json or --dataset."
            )

    sparse_metadata = _read_sparse_metadata(input_path)

    # Load dense weights and convert only floating-point dense tensors to bf16.
    dense_dict = load_dense_weights(input_path)
    _log_dtype_distribution(
        "Original ckpt dense",
        _summarize_dtype_distribution_from_tensors(dense_dict),
    )
    if sparse_metadata is not None:
        _log_dtype_distribution(
            "Original ckpt sparse",
            _summarize_dtype_distribution_from_sparse_metadata(sparse_metadata),
        )
    dense_dict_bf16 = _cast_dense_weights_to_bf16(dense_dict)

    # Build converted dict with "hstu." prefix using bf16 floating dense tensors.
    converted: Dict[str, torch.Tensor] = {}
    for k, v in dense_dict_bf16.items():
        converted["hstu." + k] = v

    has_sparse_dir = os.path.isdir(os.path.join(input_path, "sparse"))
    # Load sparse weights if requested
    include_sparse = has_sparse_dir and not dense_only

    if include_sparse:
        config_path = os.path.join(output_dir, "config.json")
        if not os.path.exists(config_path):
            logger.error(
                "Cannot load sparse weights without config.json. "
                "Provide --config-json or --dataset."
            )
        else:
            _init_single_process_distributed()
            try:
                sparse_dict = load_sparse_weights(
                    input_path,
                    output_dir,
                    dense_dict_bf16,
                    sparse_metadata=sparse_metadata,
                )
                for k, v in sparse_dict.items():
                    converted["hstu." + k] = v
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
            logger.warning("safetensors not installed, falling back to .pt format")
            out_file = os.path.join(output_dir, "model.pt")
            torch.save(converted, out_file)
    else:
        out_file = os.path.join(output_dir, "model.pt")
        torch.save(converted, out_file)

    _log_dtype_distribution(
        "Converted output",
        _summarize_dtype_distribution_from_tensors(converted),
    )

    total_params = sum(v.numel() for v in converted.values())
    sparse_count = sum(1 for k in converted if "embedding_collection" in k)
    dense_count = len(converted) - sparse_count
    logger.info(
        "Converted %d tensors (%d parameters) -> %s",
        len(converted), total_params, out_file,
    )
    logger.info(
        "  Dense: %d tensors, Sparse: %d tensors. "
        "dtype mismatches are reported; loading still follows model tensor dtype.",
        dense_count, sparse_count,
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
        choices=[
            "movielens-1m", "movielens-20m", "movielens-13b", "movielens-18b",
            "streaming-400m", "sampled-streaming-100b",
            "kuairand-1k", "debug",
        ],
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
        "--verbose", action="store_true", default=False,
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.config_json and not args.dataset:
        logger.warning(
            "Neither --config-json nor --dataset provided. "
            "config.json will NOT be generated."
        )

    convert(
        input_path=args.input_path,
        output_dir=args.output_dir,
        config_json=args.config_json,
        dataset=args.dataset,
        dense_only=args.dense_only,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
