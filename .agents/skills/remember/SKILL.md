---
name: remember
description: Explicitly save an insight, decision, or learning to agentmemory's long-term storage for Cham-OCR. Use when the user says "remember this", "ghi nhớ", "lưu lại", or when important project knowledge needs preserving.
---

# Remember Skill for Cham-OCR

This skill saves critical insights, architecture decisions, model benchmark observations, and user preferences into persistent Agent Memory for the `Cham-OCR` project.

## Workflow

1. Analyze what needs to be remembered:
   - Extract the core insight, user preference, decision, bug cause, or training parameter.
   - Choose a clear memory type: `pattern`, `preference`, `architecture`, `bug`, `workflow`, or `fact`.
   - Extract 2-5 relevant keywords/concepts (`concepts`, comma-separated string).
   - Identify relevant files (`files`, comma-separated relative paths).

2. Call the `agentmemory` MCP tool `memory_save`:
   - **ServerName**: `agentmemory`
   - **ToolName**: `memory_save`
   - **Arguments**:
     - `content`: The detailed insight/decision/instruction to remember.
     - `concepts`: Comma-separated key concepts (e.g. `"cham-ocr, v24, dataset, kaggle"`).
     - `files`: Comma-separated file paths (e.g. `"scripts/generate_data.py, .agents/AGENTS.md"`).
     - `project`: `"cham-ocr"` (always use this canonical project ID).
     - `type`: Memory type (`pattern`, `preference`, `architecture`, `bug`, `workflow`, `fact`).

3. Confirm to the user that the knowledge has been persisted, listing the ID, title, and tagged concepts.
