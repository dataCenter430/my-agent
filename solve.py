#!/usr/bin/env python3
"""
Term Challenge agent — SWE-forge patch generator.

The term-executor calls:
    solve(task_dir: str) -> str

task_dir layout:
    prompt.md        — GitHub issue / bug description
    buggy/           — the repository at the buggy commit (read-only inside solve())
    test.sh          — verification script (run by executor after patch applied)
    reference.patch  — ground truth (used by network for dedup scoring)

Required env var:
    CHUTES_API_KEY   — API key for https://llm.chutes.ai

Optional env vars:
    CHUTES_MODEL     — default: chutes/moonshotai/Kimi-K2.5-TEE
    CHUTES_API_BASE  — default: https://llm.chutes.ai/v1

Local debug:
    python solve.py /path/to/task_dir
"""

import json
import os
import re
import sys

import litellm

sys.path.insert(0, os.path.dirname(__file__))
from utils.helpers import (
    repo_tree,
    grep_keywords,
    read_test_hints,
    read_file,
    read_files_block,
    validate_patch_syntax,
    extract_diff,
    is_valid_diff,
)

litellm.set_verbose = False

# ──────────────────────────────────────────────────────────────────────────────
# Config — all values from env (required by no-hardcoding LLM review rule)
# ──────────────────────────────────────────────────────────────────────────────

def _model() -> str:
    return os.environ.get("CHUTES_MODEL", "chutes/moonshotai/Kimi-K2.5-TEE")

def _api_key() -> str:
    key = os.environ.get("CHUTES_API_KEY", "")
    if not key:
        raise RuntimeError("CHUTES_API_KEY environment variable is not set.")
    return key

def _api_base() -> str:
    return os.environ.get("CHUTES_API_BASE", "https://llm.chutes.ai/v1")

# ──────────────────────────────────────────────────────────────────────────────
# LLM wrapper
# ──────────────────────────────────────────────────────────────────────────────

def _chat(messages: list, max_tokens: int = 4096) -> str:
    resp = litellm.completion(
        model=_model(),
        messages=messages,
        api_key=_api_key(),
        api_base=_api_base(),
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

_ANALYST = """\
You are a senior software engineer triaging a bug report.

Given:
  - A bug description
  - A list of source files in the repository
  - (Optionally) a test script showing what must pass

Your job: identify the 3-8 source files most likely to contain the bug or require changes.

Think about:
  - Which modules are named in the error message or issue description?
  - Which files define the classes / functions mentioned?
  - Which test files reveal the expected behaviour?

Output ONLY a JSON array of relative paths. No prose, no explanation.
Example: ["src/parser.py", "src/utils.py", "tests/test_parser.py"]
"""

_PATCHER = """\
You are an expert software engineer producing a minimal, correct git patch.

RULES:
1. Output a unified diff that applies cleanly with:  git apply --index
   (run from inside the buggy/ directory)
2. File paths must be relative to the buggy/ repo root.
3. Change ONLY what is strictly necessary to fix the described bug.
   No reformatting, no style changes, no unrelated edits.
4. Include exactly 3 lines of unchanged context above and below every hunk.
5. Match the exact indentation and coding style of the surrounding code.
6. If a new file is required, use /dev/null as the source path.

THINK ABOUT THE ROOT CAUSE:
  - What is the underlying logic error, not just the surface symptom?
  - Is there an off-by-one error, a missing guard, a wrong type, a missing case?
  - Will the test.sh pass after your change?

OUTPUT FORMAT — the diff must be inside a ```diff block:
```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -N,M +N,M @@
 context line
-removed line
+added line
 context line
```
"""

_REFINER = """\
You are a senior engineer fixing a broken git patch.

Common problems to look for:
  - Context lines that don't match the actual file content
  - Wrong indentation (tabs vs spaces, wrong indent level)
  - Hunk offset wrong (@@ line numbers are off)
  - Logic error in the fix itself

Output ONLY the corrected diff inside a ```diff block. No explanation outside it.
"""

_FALLBACK = """\
You are an expert software engineer.

The previous patch attempt did not produce a valid unified diff.
Read the bug report and source files carefully and try again.

This time, start your entire response with:
```diff
diff --git a/...
and end with the closing ``` on its own line.
No text before or after the diff block.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

def _identify_files(issue: str, tree: str, test_hints: str) -> list:
    """Ask the LLM which files are most relevant to this issue."""
    context = f"## Bug report\n\n{issue}\n\n## Repository files\n\n```\n{tree}\n```"
    if test_hints:
        context += f"\n\n{test_hints}"
    context += "\n\nOutput a JSON array of the 3-8 most relevant file paths."

    reply = _chat(
        [{"role": "system", "content": _ANALYST}, {"role": "user", "content": context}],
        max_tokens=512,
    )
    m = re.search(r"\[.*?\]", reply, re.DOTALL)
    if m:
        try:
            files = json.loads(m.group())
            return [f.strip() for f in files if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass
    # Fallback: parse one path per line
    return [
        ln.strip().strip('"').strip("'").strip(",")
        for ln in reply.splitlines()
        if ln.strip() and not ln.strip().startswith(("[", "]", "#", "{"))
    ][:8]


def _build_context(issue: str, file_block: str, grep_hits: str, test_hints: str) -> str:
    parts = [f"## Bug report\n\n{issue}"]
    if test_hints:
        parts.append(test_hints)
    if grep_hits.strip():
        parts.append(f"## Grep results for issue keywords\n\n{grep_hits}")
    parts.append(f"## Source files (from buggy/ directory)\n\n{file_block}")
    parts.append(
        "Produce a minimal unified diff that fixes the bug described above. "
        "Paths must be relative to the buggy/ repo root. "
        "The patch must apply cleanly with `git apply --index`."
    )
    return "\n\n".join(parts)


def _generate_patch(context: str) -> str:
    return _chat(
        [{"role": "system", "content": _PATCHER}, {"role": "user", "content": context}],
        max_tokens=4096,
    )


def _refine_patch(issue: str, file_block: str, bad_patch: str) -> str:
    return _chat(
        [
            {"role": "system", "content": _REFINER},
            {
                "role": "user",
                "content": (
                    f"## Bug report\n\n{issue}\n\n"
                    f"## Source files\n\n{file_block}\n\n"
                    f"## Broken patch\n\n```diff\n{bad_patch}\n```\n\n"
                    "Fix this patch so it applies cleanly."
                ),
            },
        ],
        max_tokens=4096,
    )


def _fallback_patch(issue: str, file_block: str) -> str:
    return _chat(
        [
            {"role": "system", "content": _FALLBACK},
            {
                "role": "user",
                "content": (
                    f"## Bug report\n\n{issue}\n\n"
                    f"## Source files\n\n{file_block}"
                ),
            },
        ],
        max_tokens=4096,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def solve(task_dir: str) -> str:
    """
    Solve a SWE-forge task.

    Parameters
    ----------
    task_dir : str
        Directory containing prompt.md, buggy/, test.sh, reference.patch.

    Returns
    -------
    str
        A unified git diff that applies cleanly to buggy/ with `git apply`.
    """
    # ── 1. Read the issue and test hints ─────────────────────────────────────
    issue = read_file(os.path.join(task_dir, "prompt.md"))
    buggy_dir = os.path.join(task_dir, "buggy")
    test_hints = read_test_hints(task_dir)

    # ── 2. Explore the repository ─────────────────────────────────────────────
    tree = repo_tree(buggy_dir)
    grep_hits = grep_keywords(issue, buggy_dir)

    # ── 3. Identify which files to read ───────────────────────────────────────
    candidates = _identify_files(issue, tree, test_hints)
    existing = [f for f in candidates if os.path.isfile(os.path.join(buggy_dir, f))]
    if not existing:
        # Fallback: first Python files from the tree listing
        existing = [
            ln.strip().lstrip("./")
            for ln in tree.splitlines()
            if ln.strip().endswith(".py")
        ][:8]

    # ── 4. Read relevant files ────────────────────────────────────────────────
    file_block = read_files_block(existing, buggy_dir)
    context = _build_context(issue, file_block, grep_hits, test_hints)

    # ── 5. Generate patch (first attempt) ────────────────────────────────────
    raw = _generate_patch(context)
    patch = extract_diff(raw)

    # ── 6. Validate; refine if needed ─────────────────────────────────────────
    if not is_valid_diff(patch):
        raw2 = _refine_patch(issue, file_block, patch)
        patch = extract_diff(raw2)

    if not is_valid_diff(patch):
        raw3 = _fallback_patch(issue, file_block)
        patch = extract_diff(raw3)

    # ── 7. Dry-run git apply --check (catches context-line mismatches) ────────
    ok, err = validate_patch_syntax(patch, buggy_dir)
    if not ok and err.strip():
        # One more refinement pass with the git error message included
        raw4 = _chat(
            [
                {"role": "system", "content": _REFINER},
                {
                    "role": "user",
                    "content": (
                        f"## Bug report\n\n{issue}\n\n"
                        f"## Source files\n\n{file_block}\n\n"
                        f"## Patch (FAILED git apply --check)\n\n"
                        f"```diff\n{patch}\n```\n\n"
                        f"## git apply error\n\n```\n{err}\n```\n\n"
                        "Fix the patch so `git apply --check` passes."
                    ),
                },
            ],
            max_tokens=4096,
        )
        patch = extract_diff(raw4)

    return patch


# ──────────────────────────────────────────────────────────────────────────────
# Local debug entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    print(solve(task_directory))
