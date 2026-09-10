---
name: recall
description: Search agentmemory for past observations, decisions, patterns, and learnings for Cham-OCR. Use when the user asks "recall", "nhớ lại", "lần trước làm gì", or when needing past context.
---

# Recall Skill for Cham-OCR

This skill searches the persistent Agent Memory for past context, architectural decisions, model benchmark results, and user preferences for the `Cham-OCR` project.

## Workflow

1. Call the `agentmemory` MCP tool `memory_recall` or `memory_smart_search`:
   - **ServerName**: `agentmemory`
   - **ToolName**: `memory_recall` (or `memory_smart_search`)
   - **Arguments**:
     - `query`: The search query (e.g. keywords, file names, model versions like `v23`, `v24`, `kaggle`, `double-danda`, `stanzas`, `ocr.cham.asia`, etc.)
     - `limit`: 10

2. Focus on entries where `project` is `cham-ocr` or relevant concepts match.

3. Present results clearly:
   - **Type**: preference, pattern, fact, architecture, or bug
   - **Title & Content**: Verbatim stored insight
   - **Concepts & Files**: Tagged references
   - **Timestamp**: Date recorded
