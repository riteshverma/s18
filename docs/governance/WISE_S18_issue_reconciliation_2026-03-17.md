# Wise-AI + S18 Issue Reconciliation (Commit-Based)

Date: 2026-03-17

Scope reviewed:

- `riteshverma/s18` (local repo here)
- `wiseaihub/TSAI-EAG-Capstone` issues/commits/PRs

## Current stage

Based on commits and PR history, work is no longer at initial setup. The program is in a post-foundation integration and hardening stage:

- Core Wise-AI + S18 bridge is implemented.
- MockEHR adapter and S18-compatible tool stubs are merged in Capstone.
- CBC schema validation and payload normalization are implemented and tested in S18.
- Backlog state in Capstone is behind reality (many implementation issues still marked open).

## Cross-repo evidence matrix

| Issue | Recommendation | Evidence (Capstone) | Evidence (S18) | Rationale |
| --- | --- | --- | --- | --- |
| [#67](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/67) CBC Deterministic Threshold Finalization | Keep open, add progress comment | [PR #213](https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213) merged | [13ef59d](https://github.com/riteshverma/s18/commit/13ef59d), [0c0153e](https://github.com/riteshverma/s18/commit/0c0153e), [3bad1f4](https://github.com/riteshverma/s18/commit/3bad1f4) | CBC handling is implemented and stabilized, but issue DoD asks finalized threshold table review; keep open until explicit sign-off artifact is linked. |
| [#69](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/69) Define CBC Pydantic Model | Close now | - | [13ef59d](https://github.com/riteshverma/s18/commit/13ef59d) (`core/schemas/clinical.py`, tests) | Issue DoD is "Pydantic model validated"; commit and tests satisfy this. |
| [#73](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/73) CBC Agent Interface Contract | Keep open, add progress comment | [PR #213](https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213) | [13ef59d](https://github.com/riteshverma/s18/commit/13ef59d), [3bad1f4](https://github.com/riteshverma/s18/commit/3bad1f4) | Interface behavior exists in code paths, but issue asks finalized I/O contract doc; keep open until schema doc is linked in issue. |
| [#127](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/127) LangGraph Multi-Agent Orchestration | Close now | - | [7fc88db](https://github.com/riteshverma/s18/commit/7fc88db), [d2a6f25](https://github.com/riteshverma/s18/commit/d2a6f25), [ac92922](https://github.com/riteshverma/s18/commit/ac92922) | Orchestration and robustness are implemented and iterated in core loop/agent pipeline. |
| [#128](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/128) Tool-Calling Integration | Close now | [PR #213](https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213) ("s18-compatible ... tools") | [a44da72](https://github.com/riteshverma/s18/commit/a44da72), [ea65465](https://github.com/riteshverma/s18/commit/ea65465) | MCP routing and tool integration are in place; tests exist for MockEHR MCP. |
| [#129](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/129) RAG Pipeline Implementation | Keep open, add progress comment | - | [5561603](https://github.com/riteshverma/s18/commit/5561603), [d2a6f25](https://github.com/riteshverma/s18/commit/d2a6f25) | RAG/retrieval improvements exist, but issue-level acceptance should include explicit benchmark evidence in Capstone. |
| [#130](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/130) KB Ingestion & Vectorisation | Keep open, add progress comment | - | [5561603](https://github.com/riteshverma/s18/commit/5561603), [a44da72](https://github.com/riteshverma/s18/commit/a44da72) | Indexing assets and ingestion artifacts exist, but issue should remain open until ingestion evidence and quality checks are attached. |
| [#155](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/155) HITL Workflow | Keep open | - | - | No direct clinician dashboard + override-log completion evidence found in S18 commits. |
| [#156](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/156) Unit Testing AI agents/tool-calling | Keep open, add progress comment | [PR #213](https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213) includes tests | [a44da72](https://github.com/riteshverma/s18/commit/a44da72), [13ef59d](https://github.com/riteshverma/s18/commit/13ef59d), [0c0153e](https://github.com/riteshverma/s18/commit/0c0153e), [3bad1f4](https://github.com/riteshverma/s18/commit/3bad1f4) | Good test progress exists, but issue title suggests broader unit coverage than currently evidenced by selected tests. |
| [#202](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/202) Change Management Guidelines | Keep open, almost done | [a5bc628](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/a5bc628) and [2605d34](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/2605d34) include guideline docs and sprint review references | - | Checklist in issue body already has 2/4 done; keep open until cross-issue references and review acceptance are checked. |
| [#203](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/203) SAHI trial design alignment | Keep open | [a5bc628](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/a5bc628) baseline compliance docs | - | Planning docs exist; final trial design evidence likely not completed. |
| [#205](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/205) SAHI/BODH evidence index | Keep open, add progress comment | [a5bc628](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/a5bc628), [ce33dc0](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/ce33dc0) | - | Structure/docs are partially present; keep open until index is finalized against latest artifact set. |
| [#206](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/206) BODH benchmark plan | Keep open, add progress comment | [a5bc628](https://github.com/wiseaihub/TSAI-EAG-Capstone/commit/a5bc628) created BODH alignment plan | - | Foundational plan exists, but benchmark execution and reports are pending. |
| [#210](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/210) BODH platform access/setup | Keep open | - | - | No commit evidence of account/API setup completion found. |
| [#211](https://github.com/wiseaihub/TSAI-EAG-Capstone/issues/211) Full BODH validation | Keep open | - | - | This is a late-stage deliverable and should remain open. |

## Ready-to-paste issue comments

### Comment template: close now

```md
Progress verified across implementation repositories; closing as completed.

Evidence:
- <commit-or-pr-link-1>
- <commit-or-pr-link-2>

Completion note:
- Issue acceptance criteria are satisfied by implemented code and tests.
```

### Comment template: keep open with progress

```md
Progress update based on cross-repo reconciliation:

Implemented evidence:
- <commit-or-pr-link-1>
- <commit-or-pr-link-2>

Remaining before closure:
- <explicit missing acceptance artifact>

Keeping open until the remaining acceptance item is linked and reviewed.
```

### Suggested exact comments for immediate use

Issue `#69`:

```md
Progress verified across implementation repositories; closing as completed.

Evidence:
- https://github.com/riteshverma/s18/commit/13ef59d

Completion note:
- `core/schemas/clinical.py` and associated tests satisfy the issue DoD ("Pydantic model validated").
```

Issue `#127`:

```md
Progress verified across implementation repositories; closing as completed.

Evidence:
- https://github.com/riteshverma/s18/commit/7fc88db
- https://github.com/riteshverma/s18/commit/d2a6f25
- https://github.com/riteshverma/s18/commit/ac92922

Completion note:
- Multi-agent orchestration and robustness hardening are implemented in the active runtime path.
```

Issue `#128`:

```md
Progress verified across implementation repositories; closing as completed.

Evidence:
- https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213
- https://github.com/riteshverma/s18/commit/a44da72
- https://github.com/riteshverma/s18/commit/ea65465

Completion note:
- Tool-calling integration is implemented with MCP routing and compatible tool stubs.
```

Issue `#67`:

```md
Progress update based on cross-repo reconciliation:

Implemented evidence:
- https://github.com/riteshverma/s18/commit/13ef59d
- https://github.com/riteshverma/s18/commit/0c0153e
- https://github.com/riteshverma/s18/commit/3bad1f4

Remaining before closure:
- Link a finalized deterministic threshold table/review artifact in this issue.

Keeping open until the remaining acceptance item is linked and reviewed.
```

Issue `#156`:

```md
Progress update based on cross-repo reconciliation:

Implemented evidence:
- https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213
- https://github.com/riteshverma/s18/commit/a44da72
- https://github.com/riteshverma/s18/commit/13ef59d
- https://github.com/riteshverma/s18/commit/3bad1f4

Remaining before closure:
- Confirm broad unit-test coverage scope (all AI agents and tool-calling paths) and attach a consolidated test summary.

Keeping open until the remaining acceptance item is linked and reviewed.
```

## Batch update order (safe)

1. Close implementation-complete items first:
   - `#69`, `#127`, `#128`
2. Add progress comments (no close):
   - `#67`, `#73`, `#129`, `#130`, `#156`, `#202`, `#205`, `#206`
3. Leave later-stage work open:
   - `#155`, `#183+` compliance/demo/BODH execution stream, including `#210`, `#211`

## Optional `gh` command sequence

Run from any clone authenticated for `wiseaihub/TSAI-EAG-Capstone`.

```bash
# Close now
gh issue comment 69 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress verified across implementation repositories; closing as completed.\n\nEvidence:\n- https://github.com/riteshverma/s18/commit/13ef59d\n\nCompletion note:\n- core/schemas/clinical.py and associated tests satisfy the issue DoD (Pydantic model validated)."
gh issue close 69 --repo wiseaihub/TSAI-EAG-Capstone

gh issue comment 127 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress verified across implementation repositories; closing as completed.\n\nEvidence:\n- https://github.com/riteshverma/s18/commit/7fc88db\n- https://github.com/riteshverma/s18/commit/d2a6f25\n- https://github.com/riteshverma/s18/commit/ac92922\n\nCompletion note:\n- Multi-agent orchestration and robustness hardening are implemented in the active runtime path."
gh issue close 127 --repo wiseaihub/TSAI-EAG-Capstone

gh issue comment 128 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress verified across implementation repositories; closing as completed.\n\nEvidence:\n- https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213\n- https://github.com/riteshverma/s18/commit/a44da72\n- https://github.com/riteshverma/s18/commit/ea65465\n\nCompletion note:\n- Tool-calling integration is implemented with MCP routing and compatible tool stubs."
gh issue close 128 --repo wiseaihub/TSAI-EAG-Capstone

# Progress-only updates (do not close)
gh issue comment 67 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress update based on cross-repo reconciliation:\n\nImplemented evidence:\n- https://github.com/riteshverma/s18/commit/13ef59d\n- https://github.com/riteshverma/s18/commit/0c0153e\n- https://github.com/riteshverma/s18/commit/3bad1f4\n\nRemaining before closure:\n- Link a finalized deterministic threshold table/review artifact in this issue.\n\nKeeping open until the remaining acceptance item is linked and reviewed."

gh issue comment 73 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress update based on cross-repo reconciliation:\n\nImplemented evidence:\n- https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213\n- https://github.com/riteshverma/s18/commit/13ef59d\n- https://github.com/riteshverma/s18/commit/3bad1f4\n\nRemaining before closure:\n- Link finalized input/output contract documentation in this issue.\n\nKeeping open until the remaining acceptance item is linked and reviewed."

gh issue comment 129 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress update based on cross-repo reconciliation:\n\nImplemented evidence:\n- https://github.com/riteshverma/s18/commit/5561603\n- https://github.com/riteshverma/s18/commit/d2a6f25\n\nRemaining before closure:\n- Attach benchmarked RAG implementation evidence aligned to issue acceptance in this repo.\n\nKeeping open until the remaining acceptance item is linked and reviewed."

gh issue comment 130 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress update based on cross-repo reconciliation:\n\nImplemented evidence:\n- https://github.com/riteshverma/s18/commit/5561603\n- https://github.com/riteshverma/s18/commit/a44da72\n\nRemaining before closure:\n- Link finalized ingestion/vectorisation evidence and quality checks in this issue.\n\nKeeping open until the remaining acceptance item is linked and reviewed."

gh issue comment 156 --repo wiseaihub/TSAI-EAG-Capstone --body "Progress update based on cross-repo reconciliation:\n\nImplemented evidence:\n- https://github.com/wiseaihub/TSAI-EAG-Capstone/pull/213\n- https://github.com/riteshverma/s18/commit/a44da72\n- https://github.com/riteshverma/s18/commit/13ef59d\n- https://github.com/riteshverma/s18/commit/3bad1f4\n\nRemaining before closure:\n- Confirm broad unit-test coverage scope across all AI agents and tool-calling paths with a consolidated test summary.\n\nKeeping open until the remaining acceptance item is linked and reviewed."
```
