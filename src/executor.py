"""Robust shell execution with timeout, truncation, and error capture."""

from __future__ import annotations

import subprocess
import shlex
import os
from dataclasses import dataclass

MAX_OUTPUT_CHARS = 8000
TRUNCATION_MSG = "\n... [output truncated] ...\n"


@dataclass
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def output(self) -> str:
        combined = ""
        if self.stdout:
            combined += self.stdout
        if self.stderr:
            combined += self.stderr
        return combined

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        return self.exit_code != 0

    def has(self, *patterns: str) -> bool:
        text = self.output.lower()
        return all(p.lower() in text for p in patterns)

    def truncated_output(self, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        out = self.output
        if len(out) <= max_chars:
            return out
        half = max_chars // 2
        return out[:half] + TRUNCATION_MSG + out[-half:]


def run_shell(cmd: str, timeout: int = 60, cwd: str | None = None) -> ShellResult:
    """Execute a shell command, capturing stdout+stderr, respecting timeout."""
    env = os.environ.copy()
    env.setdefault("TERM", "xterm-256color")

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return ShellResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ShellResult(
            stdout="",
            stderr=f"Command timed out after {timeout}s: {cmd}",
            exit_code=124,
            timed_out=True,
        )
    except Exception as exc:
        return ShellResult(
            stdout="",
            stderr=f"Failed to execute command: {exc}",
            exit_code=1,
        )


def read_file(path: str, max_chars: int = 8000) -> str:
    """Read a file, truncating if too large."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... [file truncated at {max_chars} chars] ..."
        return content
    except FileNotFoundError:
        return f"[File not found: {path}]"
    except Exception as exc:
        return f"[Error reading {path}: {exc}]"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as exc:
        return f"[Error writing {path}: {exc}]"
