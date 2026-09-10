---
name: forget
description: Delete specific memories from agentmemory for privacy or correction. Use when the user says "forget this", "xóa bộ nhớ", or wants to delete a saved memory.
---

# Forget Skill for Cham-OCR

This skill safely removes outdated or incorrect memories from persistent Agent Memory.

> [!WARNING]
> This is a destructive operation. Always prompt for explicit confirmation from the user before deleting any memory record.

## Workflow

1. Search for matching memories using `memory_recall` or `memory_smart_search` with the user's query:
   - **ServerName**: `agentmemory`
   - **ToolName**: `memory_recall`
   - **Arguments**: `{ "query": "<user_query>", "limit": 10 }`

2. Show the matching memory entries (ID, title, content summary, date) and ask the user to confirm deletion.

3. Once explicitly confirmed by the user, call `memory_governance_delete`:
   - **ServerName**: `agentmemory`
   - **ToolName**: `memory_governance_delete`
   - **Arguments**:
     - `memoryIds`: Comma-separated list of memory IDs to delete (e.g. `"mem_1234, mem_5678"`).
     - `reason`: Explanation for deletion (e.g. `"User requested deletion of obsolete recipe"`).

4. Confirm deletion status back to the user.
