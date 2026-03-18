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

def _chat(messages: list, max_tokens: int = 4096, temperature: float = 0.2) -> str:
    """
    Single LLM call with error handling.
    Use temperature=0.0 for deterministic retry passes (guide: error recovery).
    """
    try:
        resp = litellm.completion(
            model=_model(),
            messages=messages,
            api_key=_api_key(),
            api_base=_api_base(),
            max_tokens=max_tokens,
            temperature=temperature,
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

# Guide: "Let the LLM Reason About Every Task" + ReAct pattern (THINK then ACT)
_ANALYST_TMPL = """\
You are a senior {lang} software engineer triaging a bug report.

Given a bug description, the repository file tree, and clues about which files
are involved, identify the 4-8 files most likely to contain the bug or need changes.

Prefer implementation files over configuration. Include test files only if the
fix is in the test itself.

Output ONLY a JSON array of relative file paths. No prose.
Example: ["src/parser.py", "src/utils.py"]
"""

# Guide: Pattern 2 (ReAct) — THOUGHT before ACTION gives measurably better fixes.
# Guide: "Think about the ROOT CAUSE" — explicitly prompted here.
_FIXER_TMPL = """\
You are an expert {lang} software engineer fixing a specific bug.

## Step 1 — THINK (required)
Before writing code, briefly reason (2-5 sentences):
  - What is the root cause?
  - Which file(s) must change and why?
  - What is the minimal, correct fix?

## Step 2 — FIX
For EACH file that needs changes, output it using this exact format:

=== FILE: relative/path/to/file ===
<complete fixed content — every single line, nothing omitted>
=== END ===

Rules for Step 2:
  - Include COMPLETE file content (not snippets).
  - Minimal change — do not reformat unrelated code.
  - Omit files that don't need changes.
  - Match original indentation exactly ({lang} style).
  - Common patterns: off-by-one, wrong type, None/null check, logic inversion,
    missing import, wrong method signature, unhandled exception.
"""

# Used for the retry pass — tells the LLM why the previous fix failed.
_REFINE_TMPL = """\
You are an expert {lang} software engineer. Your previous fix failed to apply.

The git apply error is shown below. This usually means your fixed content
doesn't exactly match the original file context — a line you referenced may
differ by whitespace, indentation, or wording.

Study the ORIGINAL files carefully, compare against your previous fix, and
produce a corrected version that applies cleanly.

Output ONLY === FILE: path === blocks (complete file content). No explanation.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Path normalization — guide: prevent silent empty patches from path mismatch
# ──────────────────────────────────────────────────────────────────────────────

def _match_path(llm_path: str, available: list) -> str:
    """
    Map an LLM-returned path to an actual key in `available`.
    Handles: ./src/foo.py, buggy/src/foo.py, foo.py (basename only)
    → src/foo.py (canonical relative path in originals dict).
    """
    path = llm_path.strip()
    # Strip common prefixes the LLM might add
    for prefix in ("./", "buggy/", "/"):
        while path.startswith(prefix):
            path = path[len(prefix):]

    if path in available:
        return path

    # Full suffix match: LLM omitted parent dirs
    matches = [p for p in available if p.endswith("/" + path) or p == path]
    if len(matches) == 1:
        return matches[0]

    # Basename match: LLM returned filename only
    base = os.path.basename(path)
    if base:
        matches = [p for p in available if os.path.basename(p) == base]
        if len(matches) == 1:
            return matches[0]

    return path  # Return as-is; difflib will just produce no diff


def _normalize_fixes(fixes: dict, originals: dict) -> dict:
    """Remap every key in `fixes` to the matching key in `originals`."""
    available = list(originals.keys())
    result = {}
    for llm_path, content in fixes.items():
        canonical = _match_path(llm_path, available)
        result[canonical] = content
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Adaptive context — guide: truncate outputs, manage context window
# ──────────────────────────────────────────────────────────────────────────────

def _chars_per_file(num_files: int) -> int:
    """
    Limit chars-per-file to keep total context bounded as selection grows.
    Fewer files → more chars each (more context for hard single-file bugs).
    """
    if num_files <= 2:
        return 16000
    if num_files <= 4:
        return 10000
    if num_files <= 6:
        return 7000
    return 5000

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
        temperature=0.1,
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
    Choose which files to read and fix using 4 signals:
      1. grep for issue keywords (all languages)
      2. grep for test.sh identifiers (most targeted)
      3. files explicitly referenced in test.sh
      4. recently changed files from git log (likely buggy)

    Returns (selected_files, combined_grep_hits).
    """
    issue_grep = grep_keywords(issue, buggy_dir)

    test_ids = test_info.get("identifiers", [])
    test_grep = grep_identifiers(test_ids, buggy_dir)
    combined_grep = (issue_grep + "\n" + test_grep).strip()

    test_file_refs = [
        f for f in test_info.get("test_files", [])
        if os.path.isfile(os.path.join(buggy_dir, f))
    ]

    recent = [
        f for f in git_recently_changed(buggy_dir)
        if os.path.isfile(os.path.join(buggy_dir, f))
    ]

    # Small repos: read everything directly, skip analyst LLM call
    if len(all_files) <= 8:
        _log(f"          small repo — reading all {len(all_files)} file(s)")
        return all_files, combined_grep

    # Large repos: build rich context then call analyst
    extra = []
    if recent:
        extra.append("## Recently changed files (likely related to bug)\n" +
                     "\n".join(recent[:6]))
    if test_file_refs:
        extra.append("## Files referenced in test.sh\n" +
                     "\n".join(test_file_refs[:5]))
    if combined_grep:
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

    for f in test_file_refs[:3] + recent[:2]:
        if f not in selected and os.path.isfile(os.path.join(buggy_dir, f)):
            selected.append(f)

    if not selected:
        selected = all_files[:8]

    return selected[:10], combined_grep


def _build_fixer_prompt(
    issue: str,
    file_pairs: list,
    test_content: str,
    grep_hits: str,
    framework: str,
) -> str:
    """Build the user message for the fixer LLM call."""
    file_sections = [
        f"=== FILE: {path} ===\n{content}\n=== END ===" for path, content in file_pairs
    ]
    msg = f"## Bug report\n\n{issue}"
    if test_content:
        msg += f"\n\n## test.sh (must pass after fix)\n```bash\n{test_content}\n```"
    if framework and framework != "unknown":
        msg += f"\n\nTest framework: **{framework}**"
    if grep_hits.strip():
        msg += f"\n\n## Key identifier locations (from grep)\n{grep_hits}"
    msg += "\n\n## Source files\n\n" + "\n\n".join(file_sections)
    msg += (
        "\n\nFix the bug. Remember: Step 1 = think (root cause), "
        "Step 2 = output === FILE: === blocks with complete content."
    )
    return msg


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


def _fix_files(
    issue: str,
    file_pairs: list,
    test_content: str,
    grep_hits: str,
    lang: str,
    framework: str,
) -> dict:
    """
    Ask the LLM to return complete fixed file contents (ReAct: THINK then FIX).
    Returns {relative_path: fixed_content}.
    """
    user_msg = _build_fixer_prompt(issue, file_pairs, test_content, grep_hits, framework)
    system = _FIXER_TMPL.format(lang=lang)
    reply = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=8192,
        temperature=0.2,
    )
    return _parse_file_sections(reply)


def _refine_fix(
    issue: str,
    originals: dict,
    prev_fixes: dict,
    apply_error: str,
    test_content: str,
    lang: str,
) -> dict:
    """
    ReAct refinement pass: feed the git apply error back to the LLM and ask it
    to produce a corrected fix.  Uses temperature=0.0 for determinism (guide:
    error recovery should be systematic, not creative).
    """
    orig_sections = "\n\n".join(
        f"=== FILE: {p} ===\n{c}\n=== END ===" for p, c in originals.items()
    )
    prev_sections = "\n\n".join(
        f"=== FILE: {p} ===\n{c}\n=== END ===" for p, c in prev_fixes.items()
    )

    user_msg = (
        f"## Bug report\n\n{issue}\n\n"
        f"## Original files (UNCHANGED source)\n\n{orig_sections}\n\n"
        f"## Your previous fix (FAILED git apply)\n\n{prev_sections}\n\n"
        f"## git apply --check error\n```\n{apply_error.strip()[:600]}\n```\n\n"
        "Carefully compare your fix against the original. Correct any lines that "
        "don't exactly match (whitespace, indentation, encoding).\n\n"
        "Output the corrected file(s) using:\n"
        "=== FILE: path ===\n<complete fixed content>\n=== END ==="
    )
    if test_content:
        user_msg = f"## test.sh\n```bash\n{test_content}\n```\n\n" + user_msg

    system = _REFINE_TMPL.format(lang=lang)
    reply = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        max_tokens=8192,
        temperature=0.0,  # Deterministic — we want exact corrections, not creativity
    )
    new_fixes = _parse_file_sections(reply)
    return new_fixes if new_fixes else prev_fixes


def _make_patch(originals: dict, fixes: dict) -> str:
    """
    Build a git-compatible unified diff using Python's difflib.
    The LLM never touches the patch format — always syntactically valid.
    """
    parts = []
    for filepath, fixed in fixes.items():
        original = originals.get(filepath, "")
        if original == fixed:
            continue

        orig_lines = original.splitlines(keepends=True)
        fix_lines = fixed.splitlines(keepends=True)

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
            f"buggy/ not found: {buggy_dir}\n"
            "Expected: task_dir/prompt.md + task_dir/buggy/ (git repo)"
        )

    # ── 2. Analyze test.sh + detect language ──────────────────────────────────
    test_info = analyze_test_sh(task_dir)
    if test_info["content"]:
        _log(f"test.sh  : framework={test_info['framework']} | "
             f"identifiers={test_info['identifiers'][:5]}")
    else:
        _log("test.sh  : not found (ok)")

    lang = detect_language(buggy_dir)
    _log(f"language : {lang}")

    # ── 3. Explore the repo (guide: always explore first) ─────────────────────
    _log("step 1/4 : exploring repository...")
    all_files = list_source_files(buggy_dir)
    tree = repo_tree(buggy_dir)
    _log(f"          {len(all_files)} source file(s) found")

    # ── 4. Smart file selection ───────────────────────────────────────────────
    _log("step 2/4 : selecting files to fix...")
    selected, grep_hits = _select_files(
        issue, buggy_dir, all_files, tree, test_info, lang
    )
    _log(f"          selected ({len(selected)}): {selected}")

    # ── 5. Read file contents (adaptive per-file limit) ───────────────────────
    char_limit = _chars_per_file(len(selected))
    originals = {}
    for path in selected:
        originals[path] = read_file(os.path.join(buggy_dir, path), max_chars=char_limit)
    _log(f"          reading files at {char_limit} chars/file limit")

    # ── 6. Ask LLM to fix (THINK then FIX — guide: ReAct pattern) ────────────
    _log("step 3/4 : asking LLM to fix files (THINK → FIX)...")
    file_pairs = list(originals.items())
    fixes = _fix_files(
        issue, file_pairs,
        test_info.get("content", ""),
        grep_hits,
        lang,
        test_info.get("framework", "unknown"),
    )

    # Hard retry if LLM returned no === FILE === blocks at all
    if not fixes:
        _log("WARNING  : no file blocks returned — retrying with explicit prompt...")
        retry_msg = (
            f"## Bug report\n\n{issue}\n\n"
            + "\n\n".join(
                f"=== FILE: {p} ===\n{c}\n=== END ===" for p, c in file_pairs
            )
            + "\n\nYou MUST output at least one changed file using "
            "=== FILE: path ===\\n<content>\\n=== END ==="
        )
        reply2 = _chat(
            [{"role": "system", "content": _FIXER_TMPL.format(lang=lang)},
             {"role": "user", "content": retry_msg}],
            max_tokens=8192,
            temperature=0.2,
        )
        fixes = _parse_file_sections(reply2)

    # Normalize LLM-returned paths → match keys in `originals` dict
    # (prevents silent empty diffs from ./src/foo.py vs src/foo.py mismatch)
    fixes = _normalize_fixes(fixes, originals)
    _log(f"          LLM changed {len(fixes)} file(s): {list(fixes.keys())}")

    # ── 7. ReAct refinement loop: build → validate → refine if needed ─────────
    # Guide: Pattern 2 (ReAct) + Pattern 6 (verify before marking done)
    _log("step 4/4 : building + validating patch (up to 3 attempts)...")
    MAX_ATTEMPTS = 3
    patch = ""
    current_fixes = fixes

    for attempt in range(MAX_ATTEMPTS):
        patch = _make_patch(originals, current_fixes)

        if not patch.strip():
            _log(f"          attempt {attempt + 1}: empty patch")
            if attempt < MAX_ATTEMPTS - 1:
                _log("          LLM returned identical content — forcing a fix...")
                forced = _fix_files(
                    issue, file_pairs,
                    test_info.get("content", ""),
                    grep_hits,
                    lang,
                    test_info.get("framework", "unknown"),
                )
                if forced:
                    current_fixes = _normalize_fixes(forced, originals)
            continue

        ok, err = validate_patch_syntax(patch, buggy_dir)
        if ok:
            _log(f"          attempt {attempt + 1}: git apply --check ✓")
            break

        _log(f"          attempt {attempt + 1}: git apply --check FAILED — "
             f"{err.strip()[:100]}")

        if attempt < MAX_ATTEMPTS - 1:
            _log(f"          refining fix (temperature=0.0)...")
            refined = _refine_fix(
                issue, originals, current_fixes, err,
                test_info.get("content", ""), lang,
            )
            current_fixes = _normalize_fixes(refined, originals)
        else:
            _log("          max attempts reached — returning best-effort patch")

    if not patch.strip():
        _log("WARNING  : could not generate a non-empty patch after all retries")
        return ""

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
