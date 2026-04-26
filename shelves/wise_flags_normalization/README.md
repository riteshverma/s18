# Shelved: Wise flags normalization (2026-03-21)

This folder preserves the **reverted** experiment that added `_normalize_wise_flags`, `_merge_wise_flag_lists`, and merged footer extraction with parsed JSON in `AgentRunner._ensure_wise_output_schema`.

**Why shelved:** Runtime showed `wise.flags` as `[]` in some Wise flows; stability was preferred over dict/list normalization until re-tested end-to-end.

## Restore on a branch

1. Create a branch from your integration branch, e.g.  
   `git checkout -b feature/wise-flags-normalization`
2. Merge `RESTORE_SNIPPET.md` into `agents/base_agent.py` (methods after `_extract_wise_from_text`, then replace the `# flags` block in `_ensure_wise_output_schema`).  
   If you saved a unified diff earlier, you can `git apply` it instead.
3. Optionally restore `docs/governance/WISE_FLAGS_CONTRACT.md` from git history.
4. **Tests:** add [`tests/test_wise_flags_schema.py`](../../tests/test_wise_flags_schema.py) (or restore from history). Run `uv run pytest tests/test_wise_flags_schema.py`. The `Dockerfile` `ci` stage currently runs `compileall` only; add pytest to that stage if you want CI to run these tests.

## Contents

- `RESTORE_SNIPPET.md` — Methods and `_ensure_wise_output_schema` flags block to re-apply.

## Status on `backup/wise-flags-shelve`

Snippet merged into [`agents/base_agent.py`](../../agents/base_agent.py); unit tests added under `tests/test_wise_flags_schema.py`.
