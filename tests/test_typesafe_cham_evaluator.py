#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Cham-OCR TypeSafe AI & JEV Evaluator
"""

import sys
import unittest

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.typesafe_cham_evaluator import (
    find_jev_executable,
    JevClient,
    ChamReviewJudge,
    ChamReleaseGateJudge,
)


class TestChamTypeSafeEvaluator(unittest.TestCase):
    """Test suite for Cham-OCR TypeSafe & JEV evaluation bridge."""

    @classmethod
    def setUpClass(cls):
        cls.jev_bin = find_jev_executable()
        if not cls.jev_bin:
            raise unittest.SkipTest("jev executable not found on system")
        cls.client = JevClient(cls.jev_bin)

    def test_line_review_judgment(self):
        judge = ChamReviewJudge(self.client)
        # Clear, high confidence line
        clean_res = judge.judge_line_review(
            transcription="ꨀꨇꨣ ꨔꨕ꩝",
            ctc_confidence=0.98,
            crop_candidates={"dbnet": "ꨀꨇꨣ ꨔꨕ꩝", "safe": "ꨀꨇꨣ ꨔꨕ꩝"},
            adjacent_overlap_px=0.0,
            has_ambiguous_cc=False,
        )
        self.assertIn("needs_review", clean_res)
        self.assertIn("review_probability", clean_res)
        self.assertLess(clean_res["review_probability"], 0.50)

    def test_release_gate_evaluation(self):
        gate_judge = ChamReleaseGateJudge(self.client)
        # Checkpoint with blank hallucinations must not be accepted cleanly
        failing_res = gate_judge.evaluate_gates(
            checkpoint_name="candidate_with_blank_hallucinations",
            baseline_cer=0.72,
            candidate_cer=1.85,
            mandatory_symbols_failing=3,
            hyphen_accuracy=81.2,
            blank_hallucinations=5,
            additional_notes="Severe blank hallucination and CER regression",
        )
        self.assertIn(failing_res["decision"], ["INVESTIGATE", "ROLLBACK"])
        self.assertLess(failing_res["probabilities"].get("ACCEPT", 0.0), 0.20)


if __name__ == "__main__":
    unittest.main()
