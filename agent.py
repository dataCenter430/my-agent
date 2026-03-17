#!/usr/bin/env python3
"""
Term Challenge agent entry point.

Usage:
    python agent.py --instruction "Your task description here"

Environment variables (at least one required):
    LLM_PROXY_URL       - Set automatically by validators during evaluation
    OPENROUTER_API_KEY  - For local testing via OpenRouter
    ANTHROPIC_API_KEY   - For local testing via Anthropic direct
    OPENAI_API_KEY      - For local testing via OpenAI direct
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow imports from src/ when running directly
sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass

from src.llm_client import LLMClient
from src.executor import run_shell, MAX_OUTPUT_CHARS


# ──────────────────────────────────────────────────────────────────────────────
# System prompt — core of agent quality
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert terminal agent that solves programming, system administration, \
and data-transformation tasks autonomously inside a Linux shell environment.

══════════════════════════════════════════════════════════
MANDATORY WORKFLOW — follow this exact order every time
══════════════════════════════════════════════════════════

STEP 1 — EXPLORE (always first)
  Run: ls -la
  If README.md exists → run: cat README.md
  Check for tests/   → run: ls tests/ or find . -name "test_*.py" -o -name "*.test.*"

STEP 2 — UNDERSTAND
  Read every file relevant to the task (cat / head / grep).
  Understand the current state BEFORE touching anything.
  If code exists, read it. If config exists, read it.

STEP 3 — PLAN (think in "thinking" field)
  Think through:
  - What exactly is asked?
  - What files need to be created/modified?
  - What edge cases exist?
  - How will I verify success?

STEP 4 — EXECUTE
  Make changes incrementally. After each significant change, verify it worked.
  Run tests if available: pytest, python -m pytest, make test, npm test, etc.
  If a command fails, diagnose the error and try an alternative.

STEP 5 — VERIFY before done
  Before setting done=true, ALWAYS verify:
  - Expected files exist and have correct content
  - Tests pass (if any exist)
  - Output matches the requirement exactly
  Only set done=true after explicit verification.

══════════════════════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════════════════════

✓ ALWAYS explore before acting — never assume the environment
✓ READ existing code before modifying it
✓ VERIFY results with explicit checks (cat file, python -c "...", run tests)
✓ HANDLE errors — if a command fails, fix it; try an alternative approach
✓ COMPLETE the full task — partial solutions score zero
✗ NEVER hardcode task-specific logic — reason generically
✗ NEVER declare done without verification

══════════════════════════════════════════════════════════
RESPONSE FORMAT — always valid JSON, no extra text
══════════════════════════════════════════════════════════

{
  "thinking": "<your step-by-step reasoning about what to do next>",
  "command": "<exact shell command to run, or empty string if done>",
  "done": false
}

When the task is complete AND verified:
{
  "thinking": "<summary of what was done and how it was verified>",
  "command": "",
  "done": true
}

IMPORTANT: The "command" field runs in bash. Use && to chain commands. \
Use heredoc or printf for multi-line file writes. \
Escape special characters properly.
"""

EXPLORATION_PROMPT = """\
You are starting a new task. Before anything else, explore the environment.
Run: ls -la && ([ -f README.md ] && cat README.md || echo "No README.md found")
"""

# ──────────────────────────────────────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────────────────────────────────────

MAX_STEPS = 80
COMMAND_TIMEOUT = 120  # seconds per shell command


def build_initial_messages(instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TASK INSTRUCTION:\n{instruction}\n\n"
                "Begin by exploring the environment (Step 1 of your workflow)."
            ),
        },
    ]


def trim_context(messages: list[dict], max_chars: int = 80_000) -> list[dict]:
    """
    Keep the system message and most recent messages within a token budget.
    When context grows too large, summarise older tool outputs.
    """
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages

    # Always keep: system (index 0), first user task (index 1), last 6 messages
    head = messages[:2]
    tail = messages[-6:]
    middle = messages[2:-6]

    # Summarize middle assistant+user pairs
    summarized_content = (
        "[... earlier steps summarized for context ...]\n"
        + "\n".join(
            f"[{m['role']}]: {m['content'][:300]}..."
            for m in middle
            if len(m["content"]) > 300
        )
    )
    summary_msg = {"role": "user", "content": summarized_content}

    trimmed = head + [summary_msg] + tail
    return trimmed


def run_agent(instruction: str) -> None:
    llm = LLMClient()
    messages = build_initial_messages(instruction)

    for step in range(MAX_STEPS):
        messages = trim_context(messages)

        try:
            reply = llm.chat(messages, max_tokens=8096, temperature=0.2)
        except Exception as exc:
            print(f"[AGENT] LLM error on step {step}: {exc}", file=sys.stderr)
            break

        messages.append({"role": "assistant", "content": reply})

        parsed = llm.extract_json(reply)
        thinking = parsed.get("thinking", "")
        command = parsed.get("command", "").strip()
        done = parsed.get("done", False)

        if thinking:
            print(f"[THINK] {thinking[:200]}", file=sys.stderr)

        if done:
            print(f"[AGENT] Task complete at step {step}.", file=sys.stderr)
            break

        if not command:
            # LLM responded with no command and not done — nudge it
            messages.append({
                "role": "user",
                "content": (
                    "You returned an empty command but done=false. "
                    "Either run a verification command or set done=true if the task is complete."
                ),
            })
            continue

        print(f"[CMD] {command[:120]}", file=sys.stderr)
        result = run_shell(command, timeout=COMMAND_TIMEOUT)
        output = result.truncated_output(MAX_OUTPUT_CHARS)

        status_note = ""
        if result.timed_out:
            status_note = f"\n[WARNING: command timed out after {COMMAND_TIMEOUT}s]"
        elif result.failed:
            status_note = f"\n[Exit code: {result.exit_code}]"

        feedback = f"Exit code: {result.exit_code}\nOutput:\n{output}{status_note}"
        messages.append({"role": "user", "content": feedback})

    else:
        print(f"[AGENT] Reached max steps ({MAX_STEPS}).", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Term Challenge agent")
    parser.add_argument("--instruction", required=True, help="Task instruction to complete")
    args = parser.parse_args()

    run_agent(args.instruction)
    print("[DONE]")


if __name__ == "__main__":
    main()
