---
name: session-history
description: Show what happened in recent past sessions on this project. Use when the user asks "what did we do last time", "session history", "lịch sử phiên", or wants an overview of previous work.
---

# Session History Skill for Cham-OCR

This skill inspects recent sessions and memory records in persistent Agent Memory.

## Workflow

1. Call the `agentmemory` MCP tool `memory_sessions`:
   - **ServerName**: `agentmemory`
   - **ToolName**: `memory_sessions`
   - **Arguments**: `{}`

2. If sessions exist, display:
   - Session ID, start time, status, observation count, and summaries.

3. Also call `memory_recall` with `query: "cham-ocr"` and `limit: 10` to display recent active long-term memories for this project.
