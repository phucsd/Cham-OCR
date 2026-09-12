#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit generated V25 Kaggle submissions for syntax, safety, and invariants."""

import os
import sys
import json
import ast
import yaml

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TRAINING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def audit():
    stages = [1, 2, 3]
    for st in stages:
        nb_path = os.path.join(TRAINING_DIR, 'output', f'kaggle_v25_stage{st}_submission', f'paddleocr_cham_v25_stage{st}.ipynb')
        meta_path = os.path.join(TRAINING_DIR, 'output', f'kaggle_v25_stage{st}_submission', 'kernel-metadata.json')
        
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        print(f"=== STAGE {st} AUDIT ===")
        assert meta['accelerator'] == 'NvidiaTeslaT4', f"Stage {st}: wrong accelerator"
        assert meta['machine_shape'] == 'NvidiaTeslaT4', f"Stage {st}: wrong machine_shape"
        print(f" • Metadata accelerator: {meta['accelerator']}")
        print(f" • Kernel sources      : {meta.get('kernel_sources')}")
        print(f" • Dataset sources     : {meta.get('dataset_sources')}")
        
        code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        print(f" • Total code cells    : {len(code_cells)}")
        
        full_nb_text = ''
        for idx, c in enumerate(code_cells):
            src = ''.join(c['source'])
            full_nb_text += src + '\n'
            
            # If cell writes a file via %%writefile
            if src.startswith('%%writefile'):
                first_line = src.splitlines()[0]
                target_filename = first_line.split()[-1]
                body = '\n'.join(src.splitlines()[1:])
                if target_filename.endswith('.py'):
                    try:
                        ast.parse(body)
                    except SyntaxError as e:
                        print(f"❌ SyntaxError in embedded file {target_filename} (Stage {st} Cell {idx+1}): {e}")
                        return 1
                elif target_filename.endswith('.json'):
                    try:
                        json.loads(body)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSONDecodeError in {target_filename} (Stage {st} Cell {idx+1}): {e}")
                        return 1
                elif target_filename.endswith('.yml') or target_filename.endswith('.yaml'):
                    try:
                        yaml.safe_load(body)
                    except Exception as e:
                        print(f"❌ YAMLError in {target_filename} (Stage {st} Cell {idx+1}): {e}")
                        return 1
                continue

            # Standard Python cell: filter jupyter magics and shell continuations
            py_lines = []
            in_multiline_shell = False
            for line in src.splitlines():
                sline = line.strip()
                if in_multiline_shell:
                    if not sline.endswith('\\'):
                        in_multiline_shell = False
                    continue
                if sline.startswith('!') or sline.startswith('%'):
                    if sline.endswith('\\'):
                        in_multiline_shell = True
                    continue
                py_lines.append(line)
                
            py_code = '\n'.join(py_lines)
            try:
                ast.parse(py_code)
            except SyntaxError as e:
                print(f"❌ SyntaxError in Stage {st} Cell {idx+1}: {e}")
                print("Code snippet:\n", py_code)
                return 1
                
        # Check invariants
        assert "os.walk('/kaggle')" not in full_nb_text, f"Stage {st} still contains os.walk('/kaggle')!"
        assert 'Global.epoch_num=40' in full_nb_text, f"Stage {st} missing Global.epoch_num=40!"
        if st == 1:
            assert 'Global.stage_end_epoch=12' in full_nb_text, "Stage 1 missing Global.stage_end_epoch=12"
            assert 'Global.pretrained_model=' in full_nb_text, "Stage 1 missing Global.pretrained_model"
            assert 'cham_v25_val_freeze.zip' in full_nb_text, "Stage 1 missing cham_v25_val_freeze.zip"
        elif st == 2:
            assert 'Global.stage_end_epoch=22' in full_nb_text, "Stage 2 missing Global.stage_end_epoch=22"
            assert 'Global.checkpoints=' in full_nb_text, "Stage 2 missing Global.checkpoints"
            assert 'cham_v25_val_freeze.zip' in full_nb_text, "Stage 2 missing cham_v25_val_freeze.zip"
        elif st == 3:
            assert 'Global.stage_end_epoch=40' in full_nb_text, "Stage 3 missing Global.stage_end_epoch=40"
            assert 'Global.checkpoints=' in full_nb_text, "Stage 3 missing Global.checkpoints"

        print(f"✅ Stage {st} PASS 100% (AST Syntax, Invariants, Deterministic Paths)\n")

    print("🎉 ALL STAGES (1, 2, 3) VERIFIED 100% VALID AND COMPLIANT!")
    return 0

if __name__ == '__main__':
    sys.exit(audit())
