# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Full-window numerical parity of actual streamed CUDA candidate kernels."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.kernels.test_rec_constrained_topk_cuda import _wide_window_inputs
from vllm_gr.ops.cuda.rec_constrained_topk import (
    _rec_constrained_topk_cuda_out_unchecked,
)
from vllm_gr.v1.worker.cuda.streaming_constrained_head import StreamingConstrainedHead

# mypy: disable-error-code="misc,no-untyped-call"
pytestmark = [
    pytest.mark.gpu_test,
    pytest.mark.l2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]


def outputs(rows: int, width: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((rows, width), dtype=torch.int32, device="cuda"),
        torch.empty((rows, width), dtype=torch.float32, device="cuda"),
        torch.empty((rows, width), dtype=torch.bool, device="cuda"),
        torch.empty((rows,), dtype=torch.int32, device="cuda"),
    )


def prepare(monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, vocab: int) -> tuple[Any, ...]:
    inputs = _wide_window_inputs(dtype, 3, vocab, 64)
    metadata = SimpleNamespace(
        vocab_size=vocab, trie_count=1, max_prefix1_degree=64, max_prefix2_degree=64
    )
    backend = SimpleNamespace(
        info=SimpleNamespace(table_metadata=metadata),
        vocab_slice=None,
        _tables=lambda: inputs[2:8],
        _first_token_offsets=None,
        _max_first_degree=64,
    )
    weight = torch.empty((vocab, 1), dtype=dtype, device="cuda")
    head = StreamingConstrainedHead(weight, backend, 3, 16)

    def project(hidden: torch.Tensor, block: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
        # Isolate collection/normalization from GEMM rounding: the projected
        # values are identical to the full-window oracle, still on the GPU.
        start = block.storage_offset()
        out.copy_(inputs[0][:, start : start + out.shape[1]])
        return out

    monkeypatch.setattr(torch, "mm", project)
    hidden = torch.empty((3, 1), dtype=dtype, device="cuda")
    return inputs, head, hidden


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocab", [2049, 8193, 24576, 65537])
@pytest.mark.parametrize("step", [0, 1, 2])
def test_streamed_candidates_match_identical_full_logits(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    vocab: int,
    step: int,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, dtype, vocab)
    reference, actual = outputs(3, 16), outputs(3, 16)
    _rec_constrained_topk_cuda_out_unchecked(
        *inputs,
        step,
        16,
        64,
        64,
        *reference,
        vocab_offset=0,
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=None,
    )
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], None, current_step=step)
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("poison", ["leading_negative_infinity", "nan", "positive_infinity"])
def test_streamed_lse_preserves_nonfinite_window_semantics(
    monkeypatch: pytest.MonkeyPatch,
    poison: str,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.float32, 8193)
    if poison == "leading_negative_infinity":
        inputs[0][:, :2048].fill_(-float("inf"))
    else:
        inputs[0][:, 1024] = float("nan") if poison == "nan" else float("inf")
    reference, actual = outputs(3, 16), outputs(3, 16)
    _rec_constrained_topk_cuda_out_unchecked(
        *inputs,
        0,
        16,
        64,
        64,
        *reference,
        vocab_offset=0,
        table_vocab_size=inputs[0].shape[1],
        out_wire_logprobs=None,
    )
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], None)
    for expected, value in zip(reference, actual):
        torch.testing.assert_close(value, expected, atol=0, rtol=0, equal_nan=True)
    if poison == "leading_negative_infinity":
        denominator = head.stats[:, 0] + torch.log(head.stats[:, 1])
        torch.testing.assert_close(
            denominator,
            torch.logsumexp(
                inputs[0] / torch.where(inputs[-1] == 0, 1.0, inputs[-1])[:, None], dim=1
            ),
            atol=2e-6,
            rtol=2e-6,
        )


def test_graph_replay_reads_updated_logits_and_step(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.bfloat16, 8193)
    actual, reference = outputs(3, 16), outputs(3, 16)
    step = torch.zeros(1, dtype=torch.int32, device="cuda")
    head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], step)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head.select(hidden, inputs[1], inputs[-1], actual[:3], actual[3], step)
    for current in (0, 1, 2):
        step.fill_(current)
        inputs[0].add_(0.25)
        graph.replay()
        _rec_constrained_topk_cuda_out_unchecked(
            *inputs,
            current,
            16,
            64,
            64,
            *reference,
            vocab_offset=0,
            table_vocab_size=inputs[0].shape[1],
            out_wire_logprobs=None,
        )
        for expected, value in zip(reference, actual):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)


def test_changed_output_geometry_is_rejected_before_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, head, hidden = prepare(monkeypatch, torch.bfloat16, 8193)
    wrong = outputs(3, 8)
    with pytest.raises(ValueError, match="output geometry changed"):
        head.select(hidden, inputs[1], inputs[-1], wrong[:3], wrong[3], None)


def _test_model(vocab: int) -> Any:
    head = torch.nn.Linear(1, vocab, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        head.weight.zero_()
        head.weight[10:18, 0].copy_(torch.arange(1, 9, device="cuda", dtype=torch.bfloat16))
    return SimpleNamespace(lm_head=head, logits_processor=SimpleNamespace(), compute_logits=head)


@torch.inference_mode()
def test_enabled_streaming_preserves_unconstrained_decode() -> None:
    from tests.kernels.beam_stage_test_utils import (
        dispatch,
        metadata,
        prepare_binding,
        runtime,
    )
    from vllm_gr.v1.beam.dispatch import GR_WORKER_RESULT_ATTR
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import _hidden_logits

    owner = runtime(
        torch.bfloat16,
        unconstrained=True,
        gr_options={"enable_streaming_constrained_head": True, "head_slice": 0},
    )
    owner.worker.model = _test_model(24576)
    for stage in range(3):
        entry = owner.admit(dispatch(metadata(stage))).entries[0]
        hidden = torch.ones((4 if stage else 1, 1), device="cuda", dtype=torch.bfloat16)
        if stage:
            prepare_binding(owner, entry)
            owner.logits = _hidden_logits(owner, hidden)
            assert owner.logits is not None and owner._stream_head is None
        else:
            owner.logits = owner.worker.model.compute_logits(hidden)
        result = getattr(owner.sample().get_output(), GR_WORKER_RESULT_ATTR)["session-0"]
    assert result.finished and result.terminal_result.output is not None


def _table_for_reload(degree: int, vocab: int) -> Any:
    import numpy as np

    from vllm_gr.constraints.table import (
        ConstraintTable,
        ConstraintTableMetadata,
        validate_constraint_table,
    )

    seconds = np.arange(10, 10 + degree, dtype=np.int32)
    arrays = {
        "first_token_ids": np.array([1], dtype=np.int32),
        "prefix1_offsets": np.r_[
            np.zeros(2, dtype=np.int32), np.full(vocab - 1, degree, dtype=np.int32)
        ],
        "prefix1_values": seconds,
        "prefix1_pair_keys": vocab + seconds.astype(np.int64),
        "prefix2_value_offsets": np.arange(degree + 1, dtype=np.int32),
        "prefix2_values": np.full(degree, vocab - 1, dtype=np.int32),
    }
    table = ConstraintTable(
        **arrays,
        metadata=ConstraintTableMetadata(
            format="constraint_table_v1",
            artifact_digest="sha256:" + f"{degree:064x}",
            tokenizer_digest="sha256:" + "a" * 64,
            vocab_size=vocab,
            max_prefix1_degree=degree,
            max_prefix2_degree=1,
            total_bytes=sum(a.nbytes for a in arrays.values()),
        ),
    )
    validate_constraint_table(table)
    return table


@torch.inference_mode()
def test_streamed_sampling_rebuilds_after_same_manager_reload() -> None:
    from tests.kernels.beam_stage_test_utils import (
        dispatch,
        metadata,
        prepare_binding,
        runtime,
    )
    from vllm_gr.constraints.loader import _upload_cuda_backend
    from vllm_gr.constraints.versioned_table import VersionedConstraintTable
    from vllm_gr.v1.worker.cuda.gpu_beam_stage_runner import _hidden_logits
    from vllm_gr.v1.worker.hot_update import reload_trie

    vocab = 24576
    owner = runtime(torch.bfloat16, gr_options={"enable_streaming_constrained_head": True})
    owner.worker.model = _test_model(vocab)

    def load(degree: int) -> Any:
        backend, _ = _upload_cuda_backend(
            _table_for_reload(degree, vocab),
            device=owner.beam.state.device,
            max_top_k=4,
            max_rows=4,
        )
        backend._mark_ready()
        return backend

    original = load(2)
    replacement = load(8)
    manager = VersionedConstraintTable(original, loader=lambda _: replacement)
    owner.backend = manager
    owner.worker.constraint_backend = manager
    owner.admit(dispatch(metadata()))
    owner.logits = owner.worker.model.compute_logits(
        torch.ones((1, 1), device="cuda", dtype=torch.bfloat16)
    )
    owner.sample().get_output()
    entry = owner.admit(dispatch(metadata(1))).entries[0]
    prepare_binding(owner, entry)
    hidden = torch.ones((4, 1), device="cuda", dtype=torch.bfloat16)
    owner.logits = _hidden_logits(owner, hidden)
    assert owner.logits is None
    old_head = owner._stream_head
    assert old_head is not None and old_head.candidate_columns == 4

    # Publish a higher-degree table after Decode Forward, but before sampling.
    assert reload_trie(SimpleNamespace(model_runner=owner.worker), "new", "B")["ready"]
    assert owner.backend is manager and manager.get() is replacement
    owner.sample().get_output()
    assert owner._stream_head is not old_head
    assert owner._stream_head is not None and owner._stream_head.candidate_columns == 8
    assert owner.tokens[0].tolist() == [17, 16, 15, 14]
    done = torch.Event()
    done.record()
    owner.release(owner.binding, consumer_done=done)
