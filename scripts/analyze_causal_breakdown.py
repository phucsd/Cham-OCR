#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True Minimum-Edit-Distance Levenshtein DP Causal Breakdown for Stage R2.1.4C
============================================================================
Replaces heuristic difflib.SequenceMatcher with exact Wagner-Fischer Levenshtein
dynamic programming alignment (100% matching run_v24_frozen_suite.py edit counts).
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classify_cham_codepoint(cp: str) -> str:
    if len(cp) != 1:
        return "UNKNOWN"
    cp_val = ord(cp)
    if cp_val == 0x0020:
        return "SPACE"
    elif 0xAA00 <= cp_val <= 0xAA28:
        return "BASE_CONSONANT"
    elif 0xAA29 <= cp_val <= 0xAA36:
        return "DEPENDENT_VOWEL_COMBINING_MARK"
    elif 0xAA40 <= cp_val <= 0xAA4D:
        return "FINAL_CONSONANT_SIGN"
    elif 0xAA50 <= cp_val <= 0xAA59:
        return "CHAM_DIGIT"
    elif 0xAA5C <= cp_val <= 0xAA5F:
        return "CHAM_PUNCTUATION_DANDA"
    return "OTHERS"


def levenshtein_alignment(gt: str, pred: str):
    """Computes exact minimum-edit-distance Levenshtein alignment operations (match, sub, del, ins)."""
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if gt[i - 1] == pred[j - 1] else 1):
            if gt[i - 1] == pred[j - 1]:
                ops.append(("match", i - 1, j - 1, gt[i - 1], pred[j - 1]))
            else:
                ops.append(("sub", i - 1, j - 1, gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, -1, gt[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1, "", pred[j - 1]))
            j -= 1
    ops.reverse()
    return dp[m][n], ops


def analyze_ops(records):
    dels, subs, ins = 0, 0, 0
    del_chars = {}
    del_cats = {}
    sub_cats = {}
    ins_cats = {}
    for rec in records:
        gt = rec.get("gt", "")
        pred = rec.get("norm_pred", rec.get("pred", ""))
        _, ops = levenshtein_alignment(gt, pred)
        for op, _, _, g_ch, p_ch in ops:
            if op == "del":
                dels += 1
                del_chars[g_ch] = del_chars.get(g_ch, 0) + 1
                cat = classify_cham_codepoint(g_ch)
                del_cats[cat] = del_cats.get(cat, 0) + 1
            elif op == "sub":
                subs += 1
                cat = classify_cham_codepoint(g_ch)
                sub_cats[cat] = sub_cats.get(cat, 0) + 1
            elif op == "ins":
                ins += 1
                cat = classify_cham_codepoint(p_ch)
                ins_cats[cat] = ins_cats.get(cat, 0) + 1
    return dels, subs, ins, del_chars, del_cats, sub_cats, ins_cats


def main():
    a_path = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", "r2_1_4c_control_continue_iter_epoch_2_raw", "benchmark_summary.json")
    b_path = os.path.join(PROJECT_ROOT, "ocr-benchmark", "results", "r2_1_4c_combmark_replay_iter_epoch_2_raw", "benchmark_summary.json")

    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))

    char_a = a["character_suite"]
    char_b = b["character_suite"]

    records_a = char_a.get("all_records", char_a.get("records", []))
    records_b = char_b.get("all_records", char_b.get("records", []))

    dels_a, subs_a, ins_a, del_chars_a, del_cats_a, sub_cats_a, ins_cats_a = analyze_ops(records_a)
    dels_b, subs_b, ins_b, del_chars_b, del_cats_b, sub_cats_b, ins_cats_b = analyze_ops(records_b)

    tot_edits_a = dels_a + subs_a + ins_a
    tot_edits_b = dels_b + subs_b + ins_b

    assert tot_edits_a == char_a["edits"], f"DP mismatch on Branch A: {tot_edits_a} != {char_a['edits']}"
    assert tot_edits_b == char_b["edits"], f"DP mismatch on Branch B: {tot_edits_b} != {char_b['edits']}"

    print(f"Branch A Char Suite (E2): total chars = {char_a['chars']}, edits = {tot_edits_a}, cer = {char_a['cer']*100:.4f}%")
    print(f"  • Deletions:     {dels_a:4d} ({dels_a/tot_edits_a*100:.2f}%) | Categories: {del_cats_a}")
    print(f"  • Substitutions: {subs_a:4d} ({subs_a/tot_edits_a*100:.2f}%) | Categories: {sub_cats_a}")
    print(f"  • Insertions:    {ins_a:4d} ({ins_a/tot_edits_a*100:.2f}%) | Categories: {ins_cats_a}")

    print(f"\nBranch B Char Suite (E2): total chars = {char_b['chars']}, edits = {tot_edits_b}, cer = {char_b['cer']*100:.4f}%")
    print(f"  • Deletions:     {dels_b:4d} ({dels_b/tot_edits_b*100:.2f}%) | Categories: {del_cats_b}")
    print(f"  • Substitutions: {subs_b:4d} ({subs_b/tot_edits_b*100:.2f}%) | Categories: {sub_cats_b}")
    print(f"  • Insertions:    {ins_b:4d} ({ins_b/tot_edits_b*100:.2f}%) | Categories: {ins_cats_b}")

    print(f"\nTrue DP Levenshtein Causal Deltas (Branch B - Branch A at E2):")
    print(f"  • Δ Total Edits:   {tot_edits_b - tot_edits_a:+4d}")
    print(f"  • Δ Deletions:     {dels_b - dels_a:+4d}")
    print(f"  • Δ Substitutions: {subs_b - subs_a:+4d}")
    print(f"  • Δ Insertions:    {ins_b - ins_a:+4d}")

    comb_marks = set([chr(c) for c in range(0xAA29, 0xAA37)] + [chr(0xAA43), chr(0xAA4C), chr(0xAA4D)])
    comb_dels_a = sum(v for k, v in del_chars_a.items() if k in comb_marks)
    comb_dels_b = sum(v for k, v in del_chars_b.items() if k in comb_marks)
    print(f"\nTarget Combining-Mark & Final-Sign Set Deletions:")
    print(f"  • Branch A: {comb_dels_a:3d} -> Branch B: {comb_dels_b:3d} (Δ = {comb_dels_b - comb_dels_a:+3d})")


if __name__ == "__main__":
    main()
