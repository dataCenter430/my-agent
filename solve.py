#!/usr/bin/env python3
"""
Term Challenge agent — SWE-forge patch generator.

The term-executor calls:
    solve(task_dir: str) -> str

task_dir layout:
    prompt.md        — GitHub issue / bug description
    buggy/           — the repository at the buggy commit
    test.sh          — verification script (must exit 0 after patch)
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
    read_file,
    read_test_hints,
    repo_tree,
    grep_keywords,
    grep_identifiers,
    list_source_files,
    detect_language,
    analyze_test_sh,
    git_recently_changed,
    validate_patch_syntax,
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
            raise RuntimeError("Chutes API error 401: invalid CHUTES_API_KEY.") from exc
        raise RuntimeError(f"LLM call failed: {exc}") from exc

# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

_ANALYST_TMPL = """\
You are a senior {lang} software engineer triaging a bug report.

Given:
  - A bug description
  - The repository file tree
  - Clues about which files are involved (grep hits, recently changed files,
    files referenced in the test script)

Identify the 4-8 files most likely to contain the bug or require changes.
Prefer implementation files over configuration files.
Include test files only if they are the direct subject of the fix.

Output ONLY a JSON array of relative file paths. No prose, no explanation.
Example: ["src/parser.py", "src/utils.py", "tests/test_parser.py"]
"""

_FIXER_TMPL = """\
You are an expert {lang} software engineer fixing a specific bug.

You will receive:
  1. A bug report
  2. The test script (test.sh) that MUST exit 0 after your fix
  3. Relevant source files

For EACH file that needs changes, output it in this exact format:

=== FILE: relative/path/to/file ===
<complete fixed content — every single line, nothing omitted>
=== END ===

Rules:
  - Output ONLY the === FILE === sections. No explanation, no commentary.
  - Include the COMPLETE file content for each changed file.
  - Make the MINIMAL change necessary — do not reformat unrelated code.
  - If a file does NOT need changes, omit it entirely.
  - Match the original indentation style exactly ({lang} style).
  - The fix must make test.sh exit 0.
  - Think about the ROOT CAUSE: off-by-one errors, wrong types, missing
    null/bounds checks, logic inversions, missing error handling.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[solve] {msg}", file=sys.stderr, flush=True)


def _identify_files(
    issue: str,
    tree: str,
    test_content: str,
    extra_context: str,
    lang: str,
) -> list:
    """Ask the LLM which files are most relevant. Returns list of relative paths."""
    ctx = f"## Bug report\n\n{issue}\n\n## Repository\n\n```\n{tree}\n```"
    if test_content:
        ctx += f"\n\n## test.sh\n```bash\n{test_content}\n```"
    if extra_context:
        ctx += f"\n\n{extra_context}"
    ctx += "\n\nOutput a JSON array of the 4-8 most relevant file paths."

    system = _ANALYST_TMPL.format(lang=lang)
    reply = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": ctx}],
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
    ][:8]


def _select_files(
    issue: str,
    buggy_dir: str,
    all_files: list,
    tree: str,
    test_info: dict,
    lang: str,
) -> tuple:
    """
    Choose which files to read and fix.
    Combines 4 signals:
      1. grep for issue keywords (all languages)
      2. grep for test.sh identifiers (function/class names being tested)
      3. files explicitly referenced in test.sh
      4. recently changed files from git log

    Returns (selected_files, combined_grep_hits).
    """
    # Signal 1: issue keyword grep
    issue_grep = grep_keywords(issue, buggy_dir)

    # Signal 2: test.sh identifier grep (most targeted signal)
    test_ids = test_info.get("identifiers", [])
    test_grep = grep_identifiers(test_ids, buggy_dir)
    combined_grep = (issue_grep + "\n" + test_grep).strip()

    # Signal 3: files referenced in test.sh
    test_file_refs = [
        f for f in test_info.get("test_files", [])
        if os.path.isfile(os.path.join(buggy_dir, f))
    ]

    # Signal 4: recently changed files
    recent = [
        f for f in git_recently_changed(buggy_dir)
        if os.path.isfile(os.path.join(buggy_dir, f))
    ]

    # For small repos: just read everything
    if len(all_files) <= 8:
        selected = all_files
        _log(f"          small repo — reading all {len(selected)} file(s)")
        return selected, combined_grep

    # For larger repos: build rich context and use analyst call
    extra = []
    if recent:
        extra.append("## Recently changed files (likely related to bug)\n" +
                     "\n".join(recent[:6]))
    if test_file_refs:
        extra.append("## Files referenced in test.sh\n" +
                     "\n".join(test_file_refs[:5]))
    if combined_grep:
        # Show just the file paths from grep, not the full content
        grep_files = list(dict.fromkeys(
            line.split(":")[0].lstrip("./")
            for line in combined_grep.splitlines()
            if ":" in line and not line.startswith("#")
        ))[:10]
        if grep_files:
            extra.append("## Files matching key identifiers\n" +
                         "\n".join(grep_files))
    if test_ids:
        extra.append("## Key identifiers from test.sh\n" + ", ".join(test_ids[:10]))

    candidates = _identify_files(
        issue, tree,
        test_info.get("content", ""),
        "\n\n".join(extra),
        lang,
    )
    selected = [f for f in candidates if os.path.isfile(os.path.join(buggy_dir, f))]

    # Always add test_file_refs and top recent files if not already included
    for f in test_file_refs[:3] + recent[:2]:
        if f not in selected and os.path.isfile(os.path.join(buggy_dir, f)):
            selected.append(f)

    if not selected:
        selected = all_files[:8]

    return selected[:10], combined_grep


def _fix_files(
    issue: str,
    files: list,
    test_content: str,
    grep_hits: str,
    lang: str,
    framework: str,
) -> dict:
    """
    Ask the LLM to return the complete fixed content for changed files.
    Returns {relative_path: fixed_content}.
    """
    file_sections = [
        f"=== FILE: {path} ===\n{content}\n=== END ===" for path, content in files
    ]

    user_msg = f"## Bug report\n\n{issue}"
    if test_content:
        user_msg += f"\n\n## test.sh (must pass after fix)\n```bash\n{test_content}\n```"
    if framework and framework != "unknown":
        user_msg += f"\n\nTest framework: **{framework}**"
    if grep_hits.strip():
        user_msg += f"\n\n## Key identifier locations\n{grep_hits}"
    user_msg += "\n\n## Source files\n\n" + "\n\n".join(file_sections)
    user_msg += (
        "\n\nFix the bug so test.sh exits 0. "
        "Output changed files using:\n"
        "=== FILE: path ===\n<complete fixed content>\n=== END ==="
    )

    system = _FIXER_TMPL.format(lang=lang)
    reply = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=8192,
    )

    return _parse_file_sections(reply)


def _parse_file_sections(reply: str) -> dict:
    """Parse === FILE: path === ... === END === blocks from LLM output."""
    result = {}
    pattern = re.compile(
        r"===\s*FILE:\s*(.+?)\s*===\s*\n(.*?)\n===\s*END\s*===",
        re.DOTALL,
    )
    for m in pattern.finditer(reply):
        filepath = m.group(1).strip()
        content = strip_fences(m.group(2))
        result[filepath] = content
    return result


def _make_patch(originals: dict, fixes: dict) -> str:
    """
    Build a git-compatible unified diff using Python's difflib.
    Always produces syntactically valid output — the LLM never touches the format.
    """
    parts = []
    for filepath, fixed in fixes.items():
        original = originals.get(filepath, "")
        if original == fixed:
            continue

        orig_lines = original.splitlines(keepends=True)
        fix_lines = fixed.splitlines(keepends=True)

        # Guarantee trailing newlines
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

    # ── 2. Analyse test.sh + detect language ──────────────────────────────────
    test_info = analyze_test_sh(task_dir)
    test_hints = read_test_hints(task_dir)
    if test_info["content"]:
        _log(f"test.sh  : found | framework={test_info['framework']} | "
             f"identifiers={test_info['identifiers'][:5]}")
    else:
        _log("test.sh  : not found (ok)")

    lang = detect_language(buggy_dir)
    _log(f"language : {lang}")

    # ── 3. Explore the repo ───────────────────────────────────────────────────
    _log("step 1/4 : exploring repository...")
    all_files = list_source_files(buggy_dir)
    tree = repo_tree(buggy_dir)
    _log(f"          {len(all_files)} source file(s) found")

    # ── 4. Smart file selection ───────────────────────────────────────────────
    _log("step 2/4 : selecting files to fix...")
    selected, grep_hits = _select_files(issue, buggy_dir, all_files, tree, test_info, lang)
    _log(f"          selected ({len(selected)}): {selected}")

    # ── 5. Read file contents ─────────────────────────────────────────────────
    originals = {}
    for path in selected:
        originals[path] = read_file(os.path.join(buggy_dir, path))

    # ── 6. Ask LLM to fix all selected files ─────────────────────────────────
    _log("step 3/4 : asking LLM to fix files...")
    file_pairs = list(originals.items())
    fixes = _fix_files(
        issue, file_pairs,
        test_info.get("content", ""),
        grep_hits,
        lang,
        test_info.get("framework", "unknown"),
    )

    # Retry if LLM returned nothing
    if not fixes:
        _log("WARNING  : no changes returned — retrying...")
        retry_msg = (
            f"## Bug report\n\n{issue}\n\n"
            + "\n\n".join(f"=== FILE: {p} ===\n{c}\n=== END ===" for p, c in file_pairs)
            + "\n\nYou MUST output fixed files using the === FILE: === END === format."
        )
        reply2 = _chat(
            [{"role": "system", "content": _FIXER_TMPL.format(lang=lang)},
             {"role": "user", "content": retry_msg}],
            max_tokens=8192,
        )
        fixes = _parse_file_sections(reply2)

    _log(f"          LLM changed {len(fixes)} file(s): {list(fixes.keys())}")

    # ── 7. Build patch with difflib (always valid) ────────────────────────────
    _log("step 4/4 : building patch with difflib...")
    patch = _make_patch(originals, fixes)

    if not patch.strip():
        _log("WARNING  : empty patch (LLM returned identical content)")
        return ""

    # ── 8. Validate with git apply --check ───────────────────────────────────
    ok, err = validate_patch_syntax(patch, buggy_dir)
    if ok:
        _log("          git apply --check ✓")
    else:
        _log(f"          git apply --check FAILED: {err.strip()[:120]}")

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
