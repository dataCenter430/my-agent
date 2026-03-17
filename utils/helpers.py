"""
Read-only repository exploration helpers.
All operations stay inside repo_dir — we never modify the buggy/ repo.
"""

import os
import re
import subprocess

MAX_OUTPUT = 6000
MAX_FILE_CHARS = 8000


def shell(cmd: str, cwd: str = ".", timeout: int = 30) -> str:
    """Run a shell command; return stdout+stderr, truncated to MAX_OUTPUT chars."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=timeout, cwd=cwd,
        )
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout}s]"
    except Exception as exc:
        out = f"[error: {exc}]"
    if len(out) > MAX_OUTPUT:
        half = MAX_OUTPUT // 2
        out = out[:half] + "\n...[truncated]...\n" + out[-half:]
    return out


def read_file(path: str, max_chars: int = MAX_FILE_CHARS) -> str:
    """Read a file with a size cap."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            return content[:max_chars] + f"\n...[truncated at {max_chars} chars]..."
        return content
    except FileNotFoundError:
        return f"[not found: {path}]"
    except Exception as exc:
        return f"[error reading: {exc}]"


def repo_tree(repo_dir: str) -> str:
    """
    Return a compact listing of source files + recent git history.
    Covers the most common source languages.
    """
    files = shell(
        "find . -type f "
        r"\( -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.go' "
        r"-o -name '*.rs' -o -name '*.java' -o -name '*.rb' "
        r"-o -name '*.c' -o -name '*.cpp' -o -name '*.h' "
        r"-o -name '*.cs' -o -name '*.php' \) "
        r"! -path '*/.git/*' ! -path '*/node_modules/*' "
        r"! -path '*/venv/*' ! -path '*/__pycache__/*' "
        r"! -path '*/dist/*' ! -path '*/build/*' "
        "| sort | head -80",
        cwd=repo_dir,
    )
    log = shell("git log --oneline -8 2>/dev/null", cwd=repo_dir)
    readme_hint = shell(
        "[ -f README.md ] && head -20 README.md || "
        "[ -f README.rst ] && head -20 README.rst || "
        "echo '[no README found]'",
        cwd=repo_dir,
    )
    return (
        f"## Source files\n{files}\n\n"
        f"## Recent commits\n{log}\n\n"
        f"## README (first 20 lines)\n{readme_hint}"
    )


def grep_keywords(prompt: str, repo_dir: str) -> str:
    """
    Extract identifiers / class/function names from the issue text
    and grep for them inside the repo.
    """
    # Pull identifiers from the first 4 lines (most specific part of the issue)
    first_lines = " ".join(prompt.splitlines()[:4])

    # Prefer CamelCase and snake_case identifiers (likely class/function names)
    camel = re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", first_lines)[:4]
    snake = re.findall(r"\b[a-z_][a-z0-9_]{4,}\b", first_lines)[:3]
    keywords = list(dict.fromkeys(camel + snake))  # deduplicate, preserve order

    results = []
    for kw in keywords[:5]:
        hit = shell(
            f"grep -rn --include='*.py' -m 4 {kw!r} . 2>/dev/null | head -12",
            cwd=repo_dir,
        )
        if hit.strip() and "[error" not in hit and "[timed" not in hit:
            results.append(f"# grep {kw!r}\n{hit}")
    return "\n".join(results)


def read_test_hints(task_dir: str) -> str:
    """
    Read the first 60 lines of test.sh so the LLM understands
    what the test expects — helps it produce the right fix.
    """
    test_sh = os.path.join(task_dir, "test.sh")
    if not os.path.isfile(test_sh):
        return ""
    content = read_file(test_sh, max_chars=2000)
    return f"## test.sh (verification script)\n```bash\n{content}\n```"


def read_files_block(paths: list, repo_dir: str) -> str:
    """Return formatted markdown blocks for a list of file paths."""
    blocks = []
    for path in paths:
        full = os.path.join(repo_dir, path)
        content = read_file(full)
        ext = os.path.splitext(path)[1].lstrip(".") or "text"
        blocks.append(f"### `{path}`\n```{ext}\n{content}\n```")
    return "\n\n".join(blocks)


def validate_patch_syntax(patch: str, repo_dir: str) -> tuple[bool, str]:
    """
    Dry-run `git apply --check` to verify the patch applies cleanly.
    Returns (ok, error_message).
    """
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as f:
        f.write(patch)
        patch_path = f.name
    try:
        result = shell(
            f"git apply --check {patch_path} 2>&1",
            cwd=repo_dir,
            timeout=10,
        )
        ok = "error" not in result.lower() and "failed" not in result.lower()
        return ok, result
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass


def extract_diff(text: str) -> str:
    """Pull a unified diff out of an LLM response."""
    text = text.strip()
    # Prefer explicit ```diff fence
    m = re.search(r"```diff\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Generic code fence whose content looks like a diff
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith(("diff --git", "---", "@@")):
            return candidate
    # Find first diff-looking line
    for i, line in enumerate(text.splitlines()):
        if line.startswith(("diff --git", "--- a/", "@@ ")):
            return "\n".join(text.splitlines()[i:]).strip()
    return text


def is_valid_diff(text: str) -> bool:
    """Heuristic: does this look like a real unified diff?"""
    return (
        bool(re.search(r"^[+-]{3}", text, re.MULTILINE))
        and ("@@" in text or "diff --git" in text)
    )
