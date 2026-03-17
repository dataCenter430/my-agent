#!/usr/bin/env python3
"""
Term Challenge — SWE-forge agent.

The term-executor calls:
    python agent.py --instruction "<github issue text>"

The agent modifies files in the current working directory (the cloned repo),
then prints [DONE]. The executor runs the test suite afterwards.

Required env vars (set on your Basilica executor deployment):
    CHUTES_API_KEY   — API key for https://llm.chutes.ai

Optional overrides:
    CHUTES_MODEL     — default: moonshotai/Kimi-K2.5-TEE
    CHUTES_API_BASE  — default: https://llm.chutes.ai/v1
"""

import argparse
import json
import os
import re
import subprocess
import sys

import litellm

litellm.set_verbose = False

# ──────────────────────────────────────────────────────────────────────────────
# Config — all values from environment (required by no-hardcoding rule)
# ──────────────────────────────────────────────────────────────────────────────

def _model() -> str:
    return os.environ.get("CHUTES_MODEL", "chutes/moonshotai/Kimi-K2.5-TEE")

def _api_key() -> str:
    key = os.environ.get("CHUTES_API_KEY", "")
    if not key:
        raise RuntimeError("CHUTES_API_KEY is not set.")
    return key

def _api_base() -> str:
    return os.environ.get("CHUTES_API_BASE", "https://llm.chutes.ai/v1")

# ──────────────────────────────────────────────────────────────────────────────
# Shell helper
# ──────────────────────────────────────────────────────────────────────────────

MAX_OUTPUT = 6000

def shell(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
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

# ──────────────────────────────────────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────────────────────────────────────

def llm(messages: list, max_tokens: int = 4096) -> str:
    resp = litellm.completion(
        model=_model(),
        messages=messages,
        api_key=_api_key(),
        api_base=_api_base(),
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""

def parse_reply(text: str) -> dict:
    """Extract JSON from LLM reply, tolerating markdown fences and prose."""
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            break
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"thinking": text, "command": "", "done": False}

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM = """\
You are an expert software engineer fixing GitHub issues autonomously.
You receive a bug report or feature request and must modify source files in the
current working directory to fix it. The test suite will be run automatically
after you finish.

══════════════════════════════════════════════
WORKFLOW — follow this order every single time
══════════════════════════════════════════════

1. EXPLORE  — always start here
   Run: ls && git log --oneline -5 && find . -name '*.py' ! -path '*/.git/*' \\
              ! -path '*/venv/*' ! -path '*/__pycache__/*' | head -40

2. READ — read files relevant to the issue before changing anything
   Use: cat <file>  or  head -n 80 <file>

3. FIND — use grep to locate the exact functions / lines involved
   Use: grep -n "keyword" <file>

4. EDIT — make the minimal targeted change
   Use: python3 -c "..." to write files programmatically, or
        a here-doc: cat > file.py << 'EOF' ... EOF

5. VERIFY — after every change, verify it worked
   Run: python3 -c "import <module>; ..." to test imports
   Run tests if present: python3 -m pytest <test_file> -x -q 2>&1 | tail -20

6. DONE — only when the fix is in place AND verified

══════════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════════

✓ Always explore before touching anything
✓ Read existing code before editing it
✓ Make minimal changes — do NOT reformat unrelated code
✓ Verify your fix with at least one command (import check or test run)
✓ Fix all errors before declaring done
✗ Never hardcode task-specific logic
✗ Never declare done without verification

══════════════════════════════════════════════
RESPONSE FORMAT — valid JSON only, no extra text
══════════════════════════════════════════════

{
  "thinking": "<reasoning about current state and next action>",
  "command": "<bash command to run, or empty string when done>",
  "done": false
}

When finished and verified:
{
  "thinking": "<summary of what was changed and how it was verified>",
  "command": "",
  "done": true
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Context management
# ──────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 80_000

def trim_messages(messages: list) -> list:
    """Keep system + first user msg + last 8 messages within token budget."""
    total = sum(len(m["content"]) for m in messages)
    if total <= MAX_CONTEXT_CHARS:
        return messages
    head = messages[:2]
    tail = messages[-8:]
    middle = messages[2:-8]
    if middle:
        summary = "[earlier steps omitted for context — continuing from recent state]\n" + \
            "\n".join(f"[{m['role']}]: {m['content'][:200]}..." for m in middle[-4:])
        return head + [{"role": "user", "content": summary}] + tail
    return head + tail

# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────

MAX_STEPS = 50
CMD_TIMEOUT = 120


def run(instruction: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"## GitHub Issue\n\n{instruction}\n\n"
                "Start by exploring the repository (Step 1 of your workflow)."
            ),
        },
    ]

    for step in range(MAX_STEPS):
        messages = trim_messages(messages)

        try:
            reply = llm(messages, max_tokens=4096)
        except Exception as exc:
            print(f"[agent] LLM error: {exc}", file=sys.stderr)
            break

        messages.append({"role": "assistant", "content": reply})
        parsed = parse_reply(reply)

        thinking = parsed.get("thinking", "")
        command = parsed.get("command", "").strip()
        done = bool(parsed.get("done", False))

        if thinking:
            print(f"[step {step}] {thinking[:160]}", file=sys.stderr)

        if done:
            print(f"[agent] done at step {step}.", file=sys.stderr)
            break

        if not command:
            messages.append({
                "role": "user",
                "content": (
                    "You returned an empty command but done=false. "
                    "Either run a verification command or set done=true."
                ),
            })
            continue

        print(f"[cmd] {command[:100]}", file=sys.stderr)
        output = shell(command, timeout=CMD_TIMEOUT)
        messages.append({
            "role": "user",
            "content": f"Output:\n{output}",
        })

    else:
        print(f"[agent] reached max steps ({MAX_STEPS}).", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True, help="GitHub issue text")
    args = parser.parse_args()
    run(args.instruction)
    print("[DONE]")


if __name__ == "__main__":
    main()
