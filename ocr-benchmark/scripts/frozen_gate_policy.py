#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cham-OCR Unified Frozen Gate Policy Module
==========================================
Canonical shared module defining:
- Dynamic threshold derivation from baseline_frozen_metrics.json
- Mandatory Release Gates (Clean, Character, Symbols, Hyphen, Blank Hallucination)
- Strict consistency assertions (P0/P1)
- Role assignment (RELEASE_QUALIFIED vs BEST_EFFORT)
- Composite score formula
"""

import os
import sys
import json
import hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)

CANDIDATE_CONFIG_PATHS = [
    os.path.join(BENCHMARK_DIR, "data", "frozen_suite", "baseline_frozen_metrics.json"),
    os.path.join(PROJECT_ROOT, "ocr-benchmark", "data", "frozen_suite", "baseline_frozen_metrics.json"),
    os.path.join(SCRIPT_DIR, "baseline_frozen_metrics.json"),
    os.path.join(PROJECT_ROOT, "config", "baseline_frozen_metrics.json"),
    os.path.join(os.getcwd(), "config", "baseline_frozen_metrics.json"),
    os.path.join(os.getcwd(), "baseline_frozen_metrics.json"),
    "/kaggle/working/config/baseline_frozen_metrics.json",
    "/kaggle/working/frozen_suite/baseline_frozen_metrics.json"
]


def find_baseline_config_path():
    for p in CANDIDATE_CONFIG_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "FATAL: baseline_frozen_metrics.json not found in any candidate locations!\n"
        "Single Source of Truth config is mandatory for benchmark reproducibility."
    )


def compute_file_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class FrozenGatePolicy:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = find_baseline_config_path()
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.raw_config = json.load(f)

        self.sha256 = compute_file_sha256(self.config_path)

        # Baseline CERs
        self.clean_baseline_cer = float(self.raw_config["clean"]["cer"])
        self.clean_tolerance = float(self.raw_config["clean"]["tolerance"])
        self.clean_threshold = self.clean_baseline_cer + self.clean_tolerance

        self.char_baseline_cer = float(self.raw_config["character"]["cer"])
        self.char_tolerance = float(self.raw_config["character"]["tolerance"])
        self.char_threshold = self.char_baseline_cer + self.char_tolerance

        # Gate targets
        self.symbol_min_acc = float(self.raw_config["symbols"].get("min_overall_accuracy", 0.99))
        self.symbol_min_class_acc = float(self.raw_config["symbols"].get("min_accuracy_per_class", 0.99))
        self.hyphen_min_acc = float(self.raw_config["hyphen"].get("min_overall_accuracy", 0.95))

        # Weights
        w = self.raw_config.get("composite_score_weights", {})
        self.weight_symbol = float(w.get("symbols", 0.40))
        self.weight_hyphen = float(w.get("hyphen", 0.30))
        self.weight_clean_cer = float(w.get("clean_cer", -0.15))
        self.weight_char_cer = float(w.get("character_cer", -0.15))

    def compute_composite_score(self, symbol_acc, hyphen_acc, clean_cer, char_cer):
        """
        Calculates Composite Score = 0.40 * Symbol + 0.30 * Hyphen - 0.15 * CleanCER - 0.15 * CharCER
        """
        return (
            self.weight_symbol * symbol_acc +
            self.weight_hyphen * hyphen_acc +
            self.weight_clean_cer * clean_cer +
            self.weight_char_cer * char_cer
        )

    def evaluate_gates(
        self,
        clean_cer,
        char_cer,
        symbol_acc,
        failing_symbols,
        hyphen_acc,
        blank_passed,
        blank_white="",
        blank_gray="",
        fullpage_cer=None
    ):
        """
        Dynamically evaluates all mandatory release gates.
        Returns:
            dict containing gate list, individual statuses, and valid_for_release boolean.
        """
        # Gate 1: Clean CER
        gate_clean_pass = bool(clean_cer <= self.clean_threshold)

        # Gate 2: Symbol Accuracy & Zero Failing Symbols
        gate_symbol_pass = bool(symbol_acc >= self.symbol_min_acc and len(failing_symbols) == 0)

        # Gate 3: Hyphen Multi-Context Accuracy
        gate_hyphen_pass = bool(hyphen_acc >= self.hyphen_min_acc)

        # Gate 4: Zero Blank Hallucination (White and Gray)
        gate_blank_pass = bool(blank_passed)

        # Gate 5: Character CER
        gate_char_pass = bool(char_cer <= self.char_threshold)

        gates = [
            {
                "gate": "Gate 1: Clean-Frozen Non-Regression",
                "baseline": f"{self.clean_baseline_cer * 100:.4f}%",
                "tolerance": f"+{self.clean_tolerance * 100:.2f}%",
                "threshold": f"{self.clean_threshold * 100:.4f}%",
                "value": f"{clean_cer * 100:.4f}%",
                "status": "PASS" if gate_clean_pass else "FAIL"
            },
            {
                "gate": "Gate 2: 14 Mandatory Symbols Accuracy (>=99.0% each)",
                "min_acc": f"{self.symbol_min_acc * 100:.1f}%",
                "value": f"{symbol_acc * 100:.4f}%",
                "failing_symbols": list(failing_symbols),
                "failing_count": len(failing_symbols),
                "status": "PASS" if gate_symbol_pass else "FAIL"
            },
            {
                "gate": "Gate 3: Hyphen Multi-Context Accuracy (>=95.0%)",
                "min_acc": f"{self.hyphen_min_acc * 100:.1f}%",
                "value": f"{hyphen_acc * 100:.4f}%",
                "status": "PASS" if gate_hyphen_pass else "FAIL"
            },
            {
                "gate": "Gate 4: Zero Blank Hallucination (White & Gray)",
                "white_pred": str(blank_white),
                "gray_pred": str(blank_gray),
                "status": "PASS" if gate_blank_pass else "FAIL"
            },
            {
                "gate": "Gate 5: Character-Frozen Non-Regression",
                "baseline": f"{self.char_baseline_cer * 100:.4f}%",
                "tolerance": f"+{self.char_tolerance * 100:.2f}%",
                "threshold": f"{self.char_threshold * 100:.4f}%",
                "value": f"{char_cer * 100:.4f}%",
                "status": "PASS" if gate_char_pass else "FAIL"
            }
        ]

        if fullpage_cer is not None:
            gate_fp_pass = bool(fullpage_cer <= 0.12)
            gates.append({
                "gate": "Gate 6: FullPage-Frozen Matched Line CER (<=12.0%)",
                "threshold": "12.0000%",
                "value": f"{fullpage_cer * 100:.4f}%",
                "status": "PASS" if gate_fp_pass else "WARN"
            })

        valid_for_release = bool(
            gate_clean_pass and
            gate_symbol_pass and
            gate_hyphen_pass and
            gate_blank_pass and
            gate_char_pass
        )

        release_status = "RELEASE_ACCEPTED" if valid_for_release else "RELEASE_REJECTED"

        return {
            "valid_for_release": valid_for_release,
            "release_status": release_status,
            "gates_passed": {
                "clean_cer": gate_clean_pass,
                "character_cer": gate_char_pass,
                "symbols": gate_symbol_pass,
                "hyphen": gate_hyphen_pass,
                "blank": gate_blank_pass
            },
            "gates": gates
        }

    @staticmethod
    def derive_failing_symbols_from_records(all_records):
        """
        Derives sorted unique list of failing symbol characters from raw per-case records.
        Each record must have 'target_symbol' and 'pass' boolean.
        """
        failed = set()
        for r in all_records:
            if not r.get("pass", True):
                failed.add(r["target_symbol"])
        return sorted(list(failed))

    @staticmethod
    def verify_symbol_consistency(summary_failing, all_records):
        """
        Enforces strict consistency assertion:
        summary.failed_symbols == derive(per_case_results where pass == false)
        Raises AssertionError if mismatch is detected.
        """
        derived = FrozenGatePolicy.derive_failing_symbols_from_records(all_records)
        assert sorted(list(summary_failing)) == derived, (
            f"FATAL P1 CONSISTENCY ERROR: Summary failing symbols {summary_failing} "
            f"does not match derived failing symbols {derived} from raw per-case records!"
        )
        return True


# Global singleton instance
_DEFAULT_POLICY = None

def get_policy(config_path=None):
    global _DEFAULT_POLICY
    if _DEFAULT_POLICY is None or config_path is not None:
        _DEFAULT_POLICY = FrozenGatePolicy(config_path)
    return _DEFAULT_POLICY
