# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

"""Cross-platform sequential smoke tests for cleanup behavior."""

from __future__ import annotations

import unittest

try:
    from tests.smoke_test_utils import smoke_harness
except ImportError:
    from smoke_test_utils import smoke_harness


class TestCrossPlatformCleanup(unittest.TestCase):

    def _run_smoke_once(self, harness, device_type: str):
        harness.set_platform(device_type)
        llm = harness.create_llm()
        outputs = llm.recommend(harness.prompt(), use_tqdm=False)
        llm.shutdown()
        return outputs

    def test_gpu_then_npu_sequence_releases_platform_state(self):
        with smoke_harness("cuda") as harness:
            gpu_outputs = self._run_smoke_once(harness, "cuda")
            npu_outputs = self._run_smoke_once(harness, "npu")

            self.assertEqual(gpu_outputs[0].item_ids, [11, 12])
            self.assertEqual(npu_outputs[0].item_ids, [11, 12])
            self.assertEqual(harness.state["active_platform"], None)
            self.assertEqual(harness.state["active_engines"], 0)

    def test_npu_then_gpu_sequence_releases_platform_state(self):
        with smoke_harness("npu") as harness:
            npu_outputs = self._run_smoke_once(harness, "npu")
            gpu_outputs = self._run_smoke_once(harness, "cuda")

            self.assertEqual(npu_outputs[0].item_ids, [11, 12])
            self.assertEqual(gpu_outputs[0].item_ids, [11, 12])
            self.assertEqual(harness.state["active_platform"], None)
            self.assertEqual(harness.state["active_engines"], 0)


if __name__ == "__main__":
    unittest.main()
