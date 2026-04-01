"""
Repository exploration helpers for Term Challenge.
All shell operations are bounded by timeouts and output caps.
Files are never written outside the task directory.
"""

import os
import re
import subprocess

MAX_OUTPUT = 6000
MAX_FILE_CHARS = 8000

# Source file extensions we care about (all languages in the dataset)
_SRC_EXTS = (
    "*.py", "*.java", "*.js", "*.jsx", "*.ts", "*.tsx",
    "*.go", "*.rs", "*.rb", "*.php", "*.c", "*.cpp", "*.h",
    "*.cs", "*.kt", "*.swift", "*.scala",
)
_FIND_SRC = (
    r"find . -type f "
    r"\( -name '*.py' -o -name '*.java' -o -name '*.js' -o -name '*.jsx' "
    r"-o -name '*.ts' -o -name '*.tsx' -o -name '*.go' -o -name '*.rs' "
    r"-o -name '*.rb' -o -name '*.php' -o -name '*.c' -o -name '*.cpp' "
    r"-o -name '*.h' -o -name '*.cs' -o -name '*.kt' -o -name '*.scala' "
    r"-o -name '*.swift' \) "
    r"! -path '*/.git/*' ! -path '*/node_modules/*' "
    r"! -path '*/venv/*' ! -path '*/__pycache__/*' "
    r"! -path '*/dist/*' ! -path '*/build/*' ! -path '*/target/*'"
)


# ──────────────────────────────────────────────────────────────────────────────
# Core utilities
# ──────────────────────────────────────────────────────────────────────────────

def shell(cmd: str, cwd: str = ".", timeout: int = 30) -> str:
    """Run a shell command; return stdout+stderr, capped at MAX_OUTPUT chars."""
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
    """Read a file with a char cap."""
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


def strip_fences(text: str) -> str:
    """Remove markdown code fences that an LLM might accidentally add."""
    text = text.strip()
    m = re.search(r"^```[^\n]*\n(.*?)```\s*$", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Repository exploration
# ──────────────────────────────────────────────────────────────────────────────

def list_source_files(repo_dir: str, max_files: int = 120) -> list:
    """Return relative paths of source files in the repo (all languages)."""
    out = shell(f"{_FIND_SRC} | sort | head -{max_files}", cwd=repo_dir)
    return [
        line.strip().lstrip("./")
        for line in out.splitlines()
        if line.strip() and not line.startswith("[")
    ]


def detect_language(repo_dir: str) -> str:
    """Detect the primary programming language by counting source file extensions."""
    out = shell(
        f"{_FIND_SRC} | sed 's/.*\\.//' | sort | uniq -c | sort -rn | head -5",
        cwd=repo_dir,
    )
    ext_map = {
        "py": "Python", "java": "Java", "js": "JavaScript",
        "ts": "TypeScript", "tsx": "TypeScript", "jsx": "JavaScript",
        "go": "Go", "rs": "Rust", "rb": "Ruby",
        "php": "PHP", "c": "C", "cpp": "C++", "cs": "C#",
        "kt": "Kotlin", "scala": "Scala", "swift": "Swift",
    }
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            lang = ext_map.get(parts[1])
            if lang:
                return lang
    return "Unknown"


def repo_tree(repo_dir: str) -> str:
    """Compact source file listing + recent git log for the LLM analyst."""
    files = shell(f"{_FIND_SRC} | sort | head -80", cwd=repo_dir)
    log = shell("git log --oneline -8 2>/dev/null", cwd=repo_dir)
    readme = shell(
        "{ [ -f README.md ] && head -15 README.md; } || "
        "{ [ -f README.rst ] && head -15 README.rst; } || "
        "echo '[no README]'",
        cwd=repo_dir,
    )
    return (
        f"## Source files\n{files}\n\n"
        f"## Recent commits\n{log}\n\n"
        f"## README (first 15 lines)\n{readme}"
    )


def git_recently_changed(repo_dir: str, n_commits: int = 8) -> list:
    """Return source files touched in the last n commits (likely buggy ones)."""
    out = shell(
        f"git log --name-only --pretty=format: -n {n_commits} 2>/dev/null "
        r"| grep -E '\.(py|java|js|jsx|ts|tsx|go|rs|rb|php|c|cpp|h|cs|kt|scala|swift)$' "
        "| sort -u | head -20",
        cwd=repo_dir,
    )
    return [f.strip() for f in out.splitlines() if f.strip() and not f.startswith("[")]


# ──────────────────────────────────────────────────────────────────────────────
# Keyword / identifier extraction
# ──────────────────────────────────────────────────────────────────────────────

def grep_keywords(prompt: str, repo_dir: str) -> str:
    """
    Extract identifiers from the first 30 lines of the issue and grep
    for them across ALL source files (not just *.py — critical for Java/JS/TS).
    """
    first_lines = " ".join(prompt.splitlines()[:30])
    camel = re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", first_lines)[:4]
    snake = re.findall(r"\b[a-z_][a-z0-9_]{4,}\b", first_lines)[:3]
    keywords = list(dict.fromkeys(camel + snake))

    results = []
    for kw in keywords[:5]:
        # Search all source languages — was previously *.py only (bug)
        hit = shell(
            f"grep -rn -m 3 {kw!r} . "
            r"! -path '*/.git/*' ! -path '*/node_modules/*' "
            r"! -path '*/venv/*' ! -path '*/__pycache__/*' "
            "2>/dev/null | head -12",
            cwd=repo_dir,
        )
        if hit.strip() and "[error" not in hit and "[timed" not in hit:
            results.append(f"# grep {kw!r}\n{hit}")
    return "\n".join(results)


def grep_identifiers(identifiers: list, repo_dir: str) -> str:
    """
    Grep for a specific list of identifiers (extracted from test.sh).
    Returns file:line hits across all languages.
    """
    if not identifiers:
        return ""
    results = []
    for ident in identifiers[:8]:
        hit = shell(
            f"grep -rn -m 3 {ident!r} . "
            r"! -path '*/.git/*' ! -path '*/node_modules/*' "
            r"! -path '*/venv/*' ! -path '*/__pycache__/*' "
            "2>/dev/null | head -10",
            cwd=repo_dir,
        )
        if hit.strip() and "[error" not in hit and "[timed" not in hit:
            results.append(f"# grep '{ident}'\n{hit}")
    return "\n".join(results)


# ──────────────────────────────────────────────────────────────────────────────
# Test.sh analysis
# ──────────────────────────────────────────────────────────────────────────────

# Words too common to be useful as grep targets
_STOP_WORDS = {
    "pytest", "jest", "maven", "gradle", "cargo", "mocha", "rspec",
    "bash", "echo", "exit", "true", "false", "then", "else", "elif",
    "done", "test", "pass", "fail", "assert", "error", "from", "import",
    "class", "void", "public", "private", "static", "return", "null",
    "none", "this", "self", "with", "open", "read", "write", "path",
    "file", "args", "argv", "main", "init", "setup", "teardown",
    "before", "after", "each", "describe", "expect", "should", "have",
}


def analyze_test_sh(task_dir: str) -> dict:
    """
    Parse test.sh to extract:
      - content: full text
      - framework: pytest | jest | maven | cargo | go | rspec | unknown
      - test_files: source/test files explicitly referenced
      - identifiers: function/class names being tested (for targeted grep)
    """
    test_sh = os.path.join(task_dir, "test.sh")
    if not os.path.isfile(test_sh):
        return {"content": "", "framework": "unknown", "test_files": [], "identifiers": []}

    content = read_file(test_sh, max_chars=3000)

    # Framework detection
    framework = "unknown"
    checks = [
        (r"python\s+-m\s+pytest|\bpytest\b", "pytest"),
        (r"\bnpm\s+test\b|\bjest\b|\bmocha\b|\bvitest\b", "jest/npm"),
        (r"\bmvn\b|\bgradle\b|\b./gradlew\b", "maven/gradle"),
        (r"\bcargo\s+test\b", "cargo"),
        (r"\bgo\s+test\b", "go"),
        (r"\brspec\b", "rspec"),
        (r"\bruby\b.*test|\bruby\b.*spec", "ruby"),
    ]
    for pattern, name in checks:
        if re.search(pattern, content):
            framework = name
            break

    # Extract file paths (e.g. tests/test_foo.py, src/Bar.java)
    file_refs = re.findall(
        r'[\w./\-]+\.(?:py|java|js|jsx|ts|tsx|go|rs|rb|php|c|cpp|h|cs|kt)\b',
        content,
    )
    test_files = list(dict.fromkeys(f.lstrip("./") for f in file_refs))

    # Extract targeted identifiers from test.sh:
    identifiers = []
    # pytest -k "test_function_name"
    identifiers += re.findall(r'-k\s+["\']([^"\']+)["\']', content)
    # pytest::TestClass::test_method → extract each part
    for m in re.finditer(r'::(\w+)', content):
        identifiers.append(m.group(1))
    # CamelCase (class/interface names)
    identifiers += [w for w in re.findall(r'\b[A-Z][a-zA-Z0-9]{2,}\b', content)
                    if w.lower() not in _STOP_WORDS]
    # snake_case function names (≥5 chars to reduce noise)
    identifiers += [w for w in re.findall(r'\b[a-z][a-z0-9_]{4,}\b', content)
                    if w.lower() not in _STOP_WORDS]

    # Deduplicate, preserve order, cap at 15
    seen = set()
    unique = []
    for x in identifiers:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    identifiers = unique[:15]

    return {
        "content": content,
        "framework": framework,
        "test_files": test_files[:8],
        "identifiers": identifiers,
    }


def read_test_hints(task_dir: str) -> str:
    """Return test.sh content as a formatted block for the LLM prompt."""
    info = analyze_test_sh(task_dir)
    if not info["content"]:
        return ""
    return f"## test.sh (verification script — must exit 0 after patch)\n```bash\n{info['content']}\n```"


# ──────────────────────────────────────────────────────────────────────────────
# Patch utilities
# ──────────────────────────────────────────────────────────────────────────────

def validate_patch_syntax(patch: str, repo_dir: str) -> tuple:
    """
    Dry-run git apply --check. Writes patch inside repo_dir (within task dir).
    Returns (ok: bool, error_message: str).
    Uses process exit code instead of string matching to avoid false positives
    from comments/messages that happen to contain the word "error".
    """
    patch_path = os.path.join(repo_dir, ".agent_validate.patch")
    try:
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(patch)
        result = subprocess.run(
            ["git", "apply", "--check", patch_path],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=10,
        )
        ok = result.returncode == 0
        err = (result.stdout + result.stderr).strip()
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "[git apply --check timed out]"
    except Exception as exc:
        return False, f"[validation error: {exc}]"
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass
