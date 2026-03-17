#!/usr/bin/env python3
"""
Term Challenge agent — SWE-forge patch generator.

The term-executor calls:
    solve(task_dir: str) -> str

task_dir layout:
    prompt.md        — GitHub issue / bug description
    buggy/           — the repository at the buggy commit
    test.sh          — verification script (run after patch applied)
    reference.patch  — ground truth (used for dedup scoring)

Required env var:
    CHUTES_API_KEY   — API key for https://llm.chutes.ai

Optional env vars:
    CHUTES_MODEL     — default: chutes/moonshotai/Kimi-K2.5-TEE
    CHUTES_API_BASE  — default: https://llm.chutes.ai/v1

Local debug:
    python solve.py /path/to/task_dir
"""

import difflib
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
    validate_patch_syntax,
    list_source_files,
    strip_fences,
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
    try:
        resp = litellm.completion(
            model=_model(),
            messages=messages,
            api_key=_api_key(),
            api_base=_api_base(),
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        msg = str(exc)
        if "402" in msg or "balance" in msg.lower() or "quota" in msg.lower():
            raise RuntimeError(
                "Chutes API error 402: account balance is $0. "
                "Top up at https://chutes.ai"
            ) from exc
        if "401" in msg or "unauthorized" in msg.lower():
            raise RuntimeError(
                "Chutes API error 401: invalid CHUTES_API_KEY."
            ) from exc
        raise RuntimeError(f"LLM call failed: {exc}") from exc

# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

_ANALYST = """\
You are a senior software engineer triaging a bug report.

Given a bug description and a list of source files, identify the 3-6 files
most likely to contain the bug or require changes.

Output ONLY a JSON array of relative file paths. No prose, no explanation.
Example: ["src/parser.py", "src/utils.py"]
"""

_FIXER = """\
You are an expert software engineer fixing a specific bug.

You will receive a bug report and the current content of one or more source files.

For EACH file that needs changes to fix the bug, output it using this exact format:

=== FILE: relative/path/to/file ===
<complete fixed content of the file — every line, nothing omitted>
=== END ===

Rules:
- Output ONLY the === FILE === sections. No explanation, no prose.
- Include the COMPLETE file content (not just the changed lines).
- Make the MINIMAL change necessary — do not reformat unrelated code.
- If a file does NOT need changes, omit it entirely.
- Match the original indentation style exactly (spaces vs tabs, indent level).
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[solve] {msg}", file=sys.stderr, flush=True)


def _identify_files(issue: str, tree: str, test_hints: str) -> list:
    """Ask the LLM which files are most relevant. Returns list of relative paths."""
    ctx = f"## Bug report\n\n{issue}\n\n## Repository files\n\n```\n{tree}\n```"
    if test_hints:
        ctx += f"\n\n{test_hints}"
    ctx += "\n\nOutput a JSON array of the 3-6 most relevant file paths."

    reply = _chat(
        [{"role": "system", "content": _ANALYST}, {"role": "user", "content": ctx}],
        max_tokens=512,
    )
    m = re.search(r"\[.*?\]", reply, re.DOTALL)
    if m:
        try:
            files = json.loads(m.group())
            return [f.strip() for f in files if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass
    return [
        ln.strip().strip('"').strip("'").strip(",")
        for ln in reply.splitlines()
        if ln.strip() and not ln.strip().startswith(("[", "]", "#", "{"))
    ][:6]


def _fix_files(issue: str, files: list, test_hints: str, grep_hits: str) -> dict:
    """
    Ask the LLM to return the fixed content for any files that need changing.
    Returns {relative_path: fixed_content}.
    """
    # Build the file sections for the prompt
    file_sections = []
    for path, content in files:
        file_sections.append(f"=== FILE: {path} ===\n{content}\n=== END ===")

    user_msg = f"## Bug report\n\n{issue}"
    if test_hints:
        user_msg += f"\n\n{test_hints}"
    if grep_hits.strip():
        user_msg += f"\n\n## Grep results\n\n{grep_hits}"
    user_msg += "\n\n## Source files\n\n" + "\n\n".join(file_sections)
    user_msg += (
        "\n\nFix the bug. For each file that needs changes, output:\n"
        "=== FILE: path ===\n<complete fixed content>\n=== END ===\n"
        "Omit files that don't need changes."
    )

    reply = _chat(
        [{"role": "system", "content": _FIXER}, {"role": "user", "content": user_msg}],
        max_tokens=8192,
    )

    # Parse === FILE: ... === END === sections
    result = {}
    pattern = re.compile(
        r"===\s*FILE:\s*(.+?)\s*===\s*\n(.*?)\n===\s*END\s*===",
        re.DOTALL,
    )
    for m in pattern.finditer(reply):
        filepath = m.group(1).strip()
        content = m.group(2)
        # Strip any accidental markdown fences inside the block
        content = strip_fences(content)
        result[filepath] = content

    return result


def _make_patch(originals: dict, fixes: dict) -> str:
    """
    Build a git-compatible unified diff using Python's difflib.
    originals: {relative_path: original_content}
    fixes:     {relative_path: fixed_content}
    Always produces a syntactically valid patch.
    """
    parts = []
    for filepath, fixed in fixes.items():
        original = originals.get(filepath, "")
        if original == fixed:
            continue

        orig_lines = original.splitlines(keepends=True)
        fix_lines = fixed.splitlines(keepends=True)

        # Guarantee trailing newline so diff tools don't complain
        if orig_lines and not orig_lines[-1].endswith("\n"):
            orig_lines[-1] += "\n"
        if fix_lines and not fix_lines[-1].endswith("\n"):
            fix_lines[-1] += "\n"

        diff = list(difflib.unified_diff(
            orig_lines, fix_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        ))
        if diff:
            parts.append(f"diff --git a/{filepath} b/{filepath}\n")
            parts.extend(diff)

    return "".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def solve(task_dir: str) -> str:
    """
    Solve a SWE-forge task.

    Parameters
    ----------
    task_dir : str
        Directory containing prompt.md, buggy/, and optionally test.sh.

    Returns
    -------
    str
        A unified git diff that applies cleanly to buggy/ with `git apply`.
    """
    # ── 1. Read task ─────────────────────────────────────────────────────────
    _log(f"task_dir : {task_dir}")
    issue = read_file(os.path.join(task_dir, "prompt.md"))
    buggy_dir = os.path.join(task_dir, "buggy")
    _log(f"prompt   : {len(issue)} chars")

    if not os.path.isdir(buggy_dir):
        raise RuntimeError(
            f"buggy/ directory not found: {buggy_dir}\n"
            "Expected layout: task_dir/prompt.md + task_dir/buggy/ (git repo)"
        )

    test_hints = read_test_hints(task_dir)
    _log("test.sh  : found" if test_hints else "test.sh  : not found (ok)")

    # ── 2. Explore the repo ───────────────────────────────────────────────────
    _log("step 1/4 : exploring repository...")
    all_files = list_source_files(buggy_dir)
    tree = repo_tree(buggy_dir)
    grep_hits = grep_keywords(issue, buggy_dir)
    _log(f"          {len(all_files)} source file(s) found")

    # ── 3. Select files to fix ────────────────────────────────────────────────
    if len(all_files) <= 6:
        # Small repo — read everything, skip the analyst call
        selected = all_files
        _log(f"step 2/4 : repo is small, reading all {len(selected)} file(s) directly")
    else:
        _log("step 2/4 : asking LLM to identify relevant files...")
        candidates = _identify_files(issue, tree, test_hints)
        selected = [f for f in candidates if os.path.isfile(os.path.join(buggy_dir, f))]
        if not selected:
            selected = all_files[:6]
        _log(f"          selected: {selected}")

    # ── 4. Read file contents ─────────────────────────────────────────────────
    originals = {}
    for path in selected:
        full = os.path.join(buggy_dir, path)
        originals[path] = read_file(full)

    # ── 5. Ask LLM to fix all files in one call ───────────────────────────────
    _log("step 3/4 : asking LLM to fix files...")
    file_pairs = list(originals.items())
    fixes = _fix_files(issue, file_pairs, test_hints, grep_hits)

    if not fixes:
        _log("WARNING  : LLM returned no changes — retrying with explicit instruction...")
        # Retry with a more direct prompt
        file_pairs_retry = [(p, c) for p, c in originals.items()]
        retry_msg = (
            f"## Bug report\n\n{issue}\n\n"
            + "\n\n".join(
                f"=== FILE: {p} ===\n{c}\n=== END ===" for p, c in file_pairs_retry
            )
            + "\n\nYou MUST output the fixed file(s) using the === FILE: === END === format."
        )
        reply2 = _chat(
            [{"role": "system", "content": _FIXER}, {"role": "user", "content": retry_msg}],
            max_tokens=8192,
        )
        pattern = re.compile(
            r"===\s*FILE:\s*(.+?)\s*===\s*\n(.*?)\n===\s*END\s*===", re.DOTALL
        )
        for m in pattern.finditer(reply2):
            fixes[m.group(1).strip()] = strip_fences(m.group(2))

    _log(f"          LLM changed {len(fixes)} file(s): {list(fixes.keys())}")

    # ── 6. Build patch with difflib (always valid syntax) ────────────────────
    _log("step 4/4 : building patch with difflib...")
    patch = _make_patch(originals, fixes)

    if not patch.strip():
        _log("WARNING  : empty patch (LLM may have returned identical content)")
        return ""

    # ── 7. Validate with git apply --check ────────────────────────────────────
    ok, err = validate_patch_syntax(patch, buggy_dir)
    if ok:
        _log("          git apply --check ✓")
    else:
        _log(f"          git apply --check failed: {err.strip()[:120]}")

    _log("done.")
    return patch


# ──────────────────────────────────────────────────────────────────────────────
# Local debug entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python solve.py <task_dir>", file=sys.stderr)
        print("  task_dir must contain: prompt.md  +  buggy/ (git repo)", file=sys.stderr)
        sys.exit(1)
    task_directory = sys.argv[1]
    try:
        result = solve(task_directory)
        print(result)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
