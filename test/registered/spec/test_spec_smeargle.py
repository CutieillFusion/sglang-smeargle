"""End-to-end test for SMEARGLE speculative decoding (chain mode).

SMEARGLE is an EAGLE-3-style draft whose single decoder layer is a Mamba2 SSM
block. This test checks two properties on a Llama-3.1-8B target:

  1. Speculative decoding is *lossless* under greedy sampling — the generated
     text is identical to the target-only baseline.
  2. The Mamba2 draft actually accelerates — the average accept length is > 1.

The draft checkpoint is ``neurips2026-smeargle/SMEARGLE`` (converted to the
``Mamba2LlamaSmeargleForCausalLM`` layout).
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "neurips2026-smeargle/SMEARGLE"

PROMPTS = [
    "The capital of France is",
    "Write a Python function to compute the factorial of n:",
    "Explain photosynthesis in one sentence:",
    "Once upon a time, in a distant kingdom,",
]


def _generate(base_url, prompt, max_new_tokens=48):
    r = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        },
    )
    r.raise_for_status()
    return r.json()


class TestSmeargleSpeculativeDecoding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:30020"
        cls.baseline_url = "http://127.0.0.1:30021"

        cls.spec_process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "SMEARGLE",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-num-steps",
                "5",
                "--disable-overlap-schedule",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.6",
            ],
        )
        cls.baseline_process = popen_launch_server(
            TARGET_MODEL,
            cls.baseline_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "triton", "--mem-fraction-static", "0.3"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.spec_process.pid)
        kill_process_tree(cls.baseline_process.pid)

    def test_lossless_and_accelerates(self):
        accept_lengths = []
        for prompt in PROMPTS:
            spec = _generate(self.base_url, prompt)
            base = _generate(self.baseline_url, prompt)
            # 1. Lossless: greedy spec output == target-only baseline.
            self.assertEqual(
                spec["text"], base["text"], msg=f"output diverged for {prompt!r}"
            )
            accept_lengths.append(spec["meta_info"]["spec_accept_length"])

        # 2. Acceleration: the draft contributes accepted tokens (> 1 per step).
        avg_accept = sum(accept_lengths) / len(accept_lengths)
        self.assertGreater(
            avg_accept, 1.0, msg=f"no speculative acceleration: {accept_lengths}"
        )


if __name__ == "__main__":
    unittest.main()
