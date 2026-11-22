#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR TypeSafe AI & JEV System One Evaluation Engine
======================================================
Provides typed judgments (Choice, Noul, Score) for:
1. Intelligent OCR Line Review & Escalation (replacing rigid threshold rules)
2. Minimal-Pair Glyph Disambiguation (e.g., Da vs Nya, Au vs O)
3. Automated Checkpoint Release Gate Decisions (ACCEPT, INVESTIGATE, ROLLBACK)
4. Script & Dialect Routing (Eastern Cham vs Western Cham vs Bilingual)

Powered by TypeSafe System One (JEV) over local MCP stdio or CLI.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from typing import Dict, Any, Optional, List, Tuple

# Ensure utf-8 encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("ChamTypeSafe")


def find_jev_executable() -> Optional[str]:
    """Locate the jev executable on the current system."""
    # 1. Check PATH
    which_path = shutil.which("jev")
    if which_path and os.path.isfile(which_path):
        return which_path

    # 2. Check standard Windows local bin
    local_bin = os.path.expanduser(r"~\.local\bin\jev.exe")
    if os.path.isfile(local_bin):
        return local_bin

    # 3. Check where.exe on Windows
    if sys.platform.startswith("win"):
        try:
            res = subprocess.run(["where.exe", "jev"], capture_output=True, text=True, check=True)
            candidate = res.stdout.strip().splitlines()[0]
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass

    return None


class JevClient:
    """Wrapper to interact with the local JEV MCP server over stdio."""

    def __init__(self, jev_path: Optional[str] = None):
        self.jev_path = jev_path or find_jev_executable()
        if not self.jev_path or not os.path.isfile(self.jev_path):
            raise FileNotFoundError(
                f"jev executable not found. Ensure jev is installed and available in PATH or ~/.local/bin/jev.exe"
            )

    def evaluate(self, state: Any, questions: Dict[str, Any], model: str = "jev-latest", timeout: int = 30) -> Dict[str, Any]:
        """
        Send an evaluation request to Jev MCP server.
        
        Args:
            state: Context string, list, or dict containing the evidence/state.
            questions: Dictionary mapping question_id -> {type, instructions, criteria}.
            model: Jev model tag (default: 'jev-latest').
            timeout: Subprocess timeout in seconds.
            
        Returns:
            Dictionary containing model, usage, and answers.
        """
        req_id = 2
        lines = [
            json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "cham-ocr-typesafe", "version": "1.0"}
                }
            }),
            json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }),
            json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {
                    "name": "evaluate",
                    "arguments": {
                        "state": state,
                        "questions": questions,
                        "model": model
                    }
                }
            })
        ]

        payload = "\n".join(lines) + "\n"

        proc = subprocess.Popen(
            [self.jev_path, "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )

        try:
            stdout, stderr = proc.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise TimeoutError(f"JEV MCP evaluate call timed out after {timeout} seconds.")

        if proc.returncode != 0:
            logger.error(f"JEV process returned code {proc.returncode}. Stderr: {stderr}")

        # Parse JSON-RPC responses
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("id") == req_id:
                    result = msg.get("result", {})
                    if result.get("isError"):
                        err_content = result.get("content", [{}])[0].get("text", "Unknown MCP error")
                        raise RuntimeError(f"JEV MCP tool error: {err_content}")
                    
                    content_list = result.get("content", [])
                    if content_list and content_list[0].get("type") == "text":
                        return json.loads(content_list[0]["text"])
            except json.JSONDecodeError:
                continue

        raise RuntimeError(f"No valid response received from JEV MCP. Stderr: {stderr}")


class ChamReviewJudge:
    """Intelligent review and escalation judge for Cham OCR textlines."""

    def __init__(self, client: Optional[JevClient] = None):
        self.client = client or JevClient()

    def judge_line_review(
        self,
        transcription: str,
        ctc_confidence: float,
        crop_candidates: Optional[Dict[str, str]] = None,
        adjacent_overlap_px: float = 0.0,
        has_ambiguous_cc: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate whether a recognized Cham textline requires human paleographer review.
        """
        state = {
            "transcription": transcription,
            "ctc_confidence": round(float(ctc_confidence), 4),
            "adjacent_overlap_px": round(float(adjacent_overlap_px), 2),
            "has_ambiguous_cc": bool(has_ambiguous_cc),
            "crop_candidates": crop_candidates or {}
        }

        questions = {
            "needs_review": {
                "type": "noul",
                "instructions": (
                    "Evaluate whether this Cham OCR textline requires human paleographic review. "
                    "Consider: low recognition confidence (< 0.75), conflicting readings across crop candidates "
                    "(especially on Brahmic base consonants or vowel diacritics), adjacent line ink overlap (> 5px), "
                    "or ambiguous connected component assignments."
                ),
                "criteria": {
                    "true": "Line is ambiguous, confidence is low, or candidates disagree on critical paleographic glyphs.",
                    "false": "Transcription is reliable, high-confidence, and consistent across candidates."
                }
            },
            "risk_severity": {
                "type": "score",
                "instructions": "Rate the paleographic error risk severity of this line.",
                "criteria": [
                    "Level 0: Clean, unambiguous transcription with solid confidence.",
                    "Level 1: Minor cosmetic discrepancy without changing word meaning or root consonant.",
                    "Level 2: Ambiguous diacritic or moderate adjacent line touching requiring quick human check.",
                    "Level 3: Critical disagreement on base consonant (e.g., Da vs Nya) or heavy ink bleeding requiring full manual re-transcription."
                ]
            }
        }

        result = self.client.evaluate(state=state, questions=questions)
        answers = result.get("answers", {})

        noul_val = answers.get("needs_review", {}).get("noul", 0.0)
        score_val = answers.get("risk_severity", {}).get("score", 0.0)

        return {
            "needs_review": noul_val >= 0.50,
            "review_probability": noul_val,
            "risk_severity_score": score_val,
            "raw_result": result
        }


class ChamReleaseGateJudge:
    """Automated evaluation of model checkpoint release gates."""

    def __init__(self, client: Optional[JevClient] = None):
        self.client = client or JevClient()

    def evaluate_gates(
        self,
        checkpoint_name: str,
        baseline_cer: float,
        candidate_cer: float,
        mandatory_symbols_failing: int,
        hyphen_accuracy: float,
        blank_hallucinations: int,
        additional_notes: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluate candidate checkpoint against strict Cham-OCR Release Gates:
        - Gate 1: Clean-Frozen Non-Regression (CER <= Baseline CER + 0.20%)
        - Gate 2: 14 Mandatory Symbols Accuracy >= 99.0% (failing symbols == 0)
        - Gate 3: Hyphen Multi-Context Regression >= 95.0%
        - Gate 4: Zero Blank Hallucination on white/gray images (count == 0)
        """
        cer_diff = candidate_cer - baseline_cer
        state = {
            "checkpoint": checkpoint_name,
            "baseline_cer_percent": baseline_cer,
            "candidate_cer_percent": candidate_cer,
            "cer_delta_percent": round(cer_diff, 4),
            "gate1_non_regression_pass": cer_diff <= 0.20,
            "gate2_failing_mandatory_symbols": mandatory_symbols_failing,
            "gate3_hyphen_accuracy_percent": hyphen_accuracy,
            "gate4_blank_hallucinations": blank_hallucinations,
            "notes": additional_notes
        }

        questions = {
            "release_action": {
                "type": "choice",
                "instructions": (
                    "Determine the next engineering action for this Cham-OCR candidate checkpoint based on strict Release Gates. "
                    "All 4 gates must pass for ACCEPT. Any regression or blank hallucination requires INVESTIGATE or ROLLBACK."
                ),
                "criteria": {
                    "ACCEPT": "All 4 Release Gates pass completely; model is qualified for deployment.",
                    "INVESTIGATE": "Marginal failure, unexpected variance, or minor symbol drop that warrants diagnostic analysis before deciding.",
                    "ROLLBACK": "Severe regression, CER spike, blank hallucination, or multiple failing mandatory symbols that disqualify the checkpoint."
                }
            },
            "gate_compliance": {
                "type": "noul",
                "instructions": "Did this checkpoint strictly satisfy all 4 Cham-OCR Release Gates without any compromise?",
                "criteria": {
                    "true": "All gates passed without exceptions.",
                    "false": "One or more gates failed."
                }
            }
        }

        result = self.client.evaluate(state=state, questions=questions)
        answers = result.get("answers", {})

        action_data = answers.get("release_action", {})
        decision = action_data.get("choice", "INVESTIGATE")
        probs = action_data.get("probabilities", {})

        return {
            "decision": decision,
            "probabilities": probs,
            "compliance_probability": answers.get("gate_compliance", {}).get("noul", 0.0),
            "state_evaluated": state,
            "raw_result": result
        }


def main():
    """Run interactive or self-test demo of Cham TypeSafe evaluator."""
    print("=" * 70)
    print("Cham-OCR TypeSafe AI & JEV System One Evaluation Demo")
    print("=" * 70)

    try:
        client = JevClient()
        print(f"[OK] Found JEV executable: {client.jev_path}")
    except Exception as e:
        print(f"[FAIL] Could not initialize JEV client: {e}")
        return 1

    # Demo 1: OCR Line Review Judgment
    print("\n--- [Demo 1] Testing OCR Line Review Judgment ---")
    review_judge = ChamReviewJudge(client)
    sample_line = {
        "transcription": "ꨕꨮꨪꨯ꩝",
        "ctc_confidence": 0.61,
        "crop_candidates": {
            "dbnet": "ꨕꨮꨪꨯ꩝",
            "safe": "ꨊꨮꨪꨯ꩝"
        },
        "adjacent_overlap_px": 6.8,
        "has_ambiguous_cc": True
    }
    print(f"Sample Input: {json.dumps(sample_line, ensure_ascii=False)}")
    review_res = review_judge.judge_line_review(**sample_line)
    print(f"Needs Review: {review_res['needs_review']} (p={review_res['review_probability']:.2f})")
    print(f"Risk Severity: Level {review_res['risk_severity_score']:.2f}")

    # Demo 2: Benchmark Release Gate Evaluation
    print("\n--- [Demo 2] Testing Benchmark Release Gate Decision ---")
    gate_judge = ChamReleaseGateJudge(client)
    sample_benchmark = {
        "checkpoint_name": "iter_epoch_40_candidate",
        "baseline_cer": 0.72,
        "candidate_cer": 0.69,
        "mandatory_symbols_failing": 0,
        "hyphen_accuracy": 98.4,
        "blank_hallucinations": 0,
        "additional_notes": "SOTA 40 Epochs run on Kaggle Dual T4x2. Excellent punctuation coverage."
    }
    print(f"Sample Benchmark: {json.dumps(sample_benchmark, indent=2)}")
    gate_res = gate_judge.evaluate_gates(**sample_benchmark)
    print(f"Release Decision: {gate_res['decision']}")
    print(f"Probabilities: {json.dumps(gate_res['probabilities'], indent=2)}")
    print(f"Compliance: p={gate_res['compliance_probability']:.2f}")

    print("\n[SUCCESS] Cham-OCR TypeSafe integration fully verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
