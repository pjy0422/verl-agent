#!/usr/bin/env python3
"""Visualize best/worst rollout samples from JSONL files.

Usage:
    python view_rollout.py <step>                          # default experiment
    python view_rollout.py <step> --exp <experiment_name>  # specific experiment
    python view_rollout.py <step> --dir <path/to/dir>      # direct path
    python view_rollout.py --list                           # list available steps
    python view_rollout.py <step> --top 5                   # show top-5 and bottom-5
    python view_rollout.py <step> --all                     # show all samples sorted
"""

import argparse
import json
import os
import re
import sys
import textwrap

# ── ANSI Colors ──────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_GREEN= "\033[42m"
    BG_BLUE = "\033[44m"
    GREY    = "\033[90m"


try:
    TERM_WIDTH = min(os.get_terminal_size().columns, 120)
except OSError:
    TERM_WIDTH = 120


def bar(value, max_val=1.0, width=20):
    """Render a horizontal bar."""
    ratio = min(max(value / max_val, 0), 1.0)
    filled = int(ratio * width)
    if ratio >= 0.8:
        color = C.GREEN
    elif ratio >= 0.4:
        color = C.YELLOW
    else:
        color = C.RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def extract_behavior(input_text):
    """Extract harmful behavior description from the prompt."""
    m = re.search(r"Harmful Behavior:\*\*\s*'([^']+)'", input_text)
    if m:
        return m.group(1)
    m = re.search(r"harmful behavior[:\s]+(.+)", input_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:120]
    return "(unknown behavior)"


def parse_output(output_text):
    """Parse JSON or XML formatted output into rationale + prompt."""
    # Try JSON
    try:
        obj = json.loads(output_text)
        return obj.get("rationale", ""), obj.get("prompt", obj.get("answer", ""))
    except (json.JSONDecodeError, TypeError):
        pass

    # Try XML <reasoning>/<answer>
    think_m = re.search(r"<reasoning>(.*?)</reasoning>", output_text, re.DOTALL)
    answer_m = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    if think_m or answer_m:
        return (think_m.group(1).strip() if think_m else ""), \
               (answer_m.group(1).strip() if answer_m else "")

    # Raw text fallback
    return "", output_text


def wrap(text, indent=4, width=None):
    """Word-wrap text with indent."""
    w = (width or TERM_WIDTH) - indent
    lines = textwrap.fill(text, width=w).split("\n")
    pad = " " * indent
    return "\n".join(pad + l for l in lines)


def print_separator(char="─", label=""):
    if label:
        side = (TERM_WIDTH - len(label) - 4) // 2
        print(f"{C.DIM}{char * side}{C.RESET} {C.BOLD}{label}{C.RESET} {C.DIM}{char * side}{C.RESET}")
    else:
        print(f"{C.DIM}{char * TERM_WIDTH}{C.RESET}")


def print_entry(entry, rank_label, color):
    """Pretty-print a single rollout entry."""
    behavior = extract_behavior(entry["input"])
    score = entry["score"]
    turn_rewards = entry.get("turn_rewards", [])
    turn_advantages = entry.get("turn_advantages", [])
    rationale, prompt = parse_output(entry["output"])
    num_turns = len(turn_rewards)

    # ── Header ──
    print()
    print(f"  {color}{C.BOLD}{rank_label}{C.RESET}")
    print_separator("─")

    # ── Score + Behavior ──
    print(f"  {C.BOLD}Score:{C.RESET}    {score:.4f}  {bar(score, 5.0, 30)}")
    print(f"  {C.BOLD}Behavior:{C.RESET} {C.CYAN}{behavior}{C.RESET}")
    print(f"  {C.BOLD}Turns:{C.RESET}    {num_turns}")

    # ── Per-Turn Rewards ──
    if turn_rewards:
        print()
        print(f"  {C.BOLD}Turn Rewards:{C.RESET}")
        for i, r in enumerate(turn_rewards):
            print(f"    Turn {i+1}: {r:+.4f}  {bar(r, 1.0, 20)}")

    # ── Per-Turn Advantages ──
    if turn_advantages:
        print(f"  {C.BOLD}Turn Advantages:{C.RESET}")
        adv_max = max(abs(a) for a in turn_advantages) or 1.0
        for i, a in enumerate(turn_advantages):
            sign_color = C.GREEN if a > 0 else C.RED
            print(f"    Turn {i+1}: {sign_color}{a:+.4f}{C.RESET}")

    # ── Rationale ──
    if rationale:
        print()
        print(f"  {C.BOLD}Rationale:{C.RESET}")
        print(f"{C.DIM}{wrap(rationale)}{C.RESET}")

    # ── Prompt/Action ──
    if prompt:
        print()
        print(f"  {C.BOLD}Prompt → Target:{C.RESET}")
        print(f"{C.WHITE}{wrap(prompt)}{C.RESET}")

    # ── Raw output (truncated) if parsing failed ──
    if not rationale and not prompt:
        print()
        print(f"  {C.BOLD}Raw Output:{C.RESET}")
        print(f"{C.DIM}{wrap(entry['output'][:600])}{C.RESET}")

    print_separator("─")


def main():
    parser = argparse.ArgumentParser(description="Visualize rollout JSONL data")
    parser.add_argument("step", nargs="?", type=int, help="Training step number")
    parser.add_argument("--exp", default="multiturn_grpo_qwen3_4b_4gpu",
                        help="Experiment name (default: multiturn_grpo_qwen3_4b_2gpu)")
    parser.add_argument("--dir", default=None,
                        help="Direct path to train_rollout directory")
    parser.add_argument("--list", action="store_true", help="List available steps")
    parser.add_argument("--top", type=int, default=1,
                        help="Show top-N and bottom-N (default: 1)")
    parser.add_argument("--all", action="store_true",
                        help="Show all samples sorted by score")
    parser.add_argument("--stats", action="store_true",
                        help="Show score distribution statistics only")
    args = parser.parse_args()

    # Resolve directory
    if args.dir:
        rollout_dir = args.dir
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        rollout_dir = os.path.join(base, "run_logs", args.exp, "train_rollout")

    if not os.path.isdir(rollout_dir):
        print(f"{C.RED}Error: Directory not found: {rollout_dir}{C.RESET}")
        # List available experiments
        run_logs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_logs")
        if os.path.isdir(run_logs):
            exps = [d for d in os.listdir(run_logs)
                    if os.path.isdir(os.path.join(run_logs, d, "train_rollout"))]
            if exps:
                print(f"\n{C.BOLD}Available experiments:{C.RESET}")
                for e in sorted(exps):
                    print(f"  --exp {e}")
        sys.exit(1)

    # Get available steps
    steps = sorted(int(f.replace(".jsonl", ""))
                   for f in os.listdir(rollout_dir) if f.endswith(".jsonl"))

    if not steps:
        print(f"{C.RED}No JSONL files found in {rollout_dir}{C.RESET}")
        sys.exit(1)

    if args.list or args.step is None:
        print(f"\n{C.BOLD}Experiment:{C.RESET} {args.exp}")
        print(f"{C.BOLD}Directory:{C.RESET}  {rollout_dir}")
        print(f"{C.BOLD}Available steps ({len(steps)}):{C.RESET}")
        # Print in columns
        cols = TERM_WIDTH // 8
        for i in range(0, len(steps), cols):
            row = steps[i:i+cols]
            print("  " + "".join(f"{s:>6}" for s in row))
        print(f"\n{C.DIM}Usage: python view_rollout.py <step>{C.RESET}")
        sys.exit(0)

    step = args.step
    jsonl_path = os.path.join(rollout_dir, f"{step}.jsonl")
    if not os.path.exists(jsonl_path):
        closest = min(steps, key=lambda s: abs(s - step))
        print(f"{C.RED}Step {step} not found.{C.RESET} Closest: {closest}")
        sys.exit(1)

    # Load data
    with open(jsonl_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    entries.sort(key=lambda x: x["score"])
    scores = [e["score"] for e in entries]

    # ── Header ──
    print()
    print_separator("═", f"Step {step}")
    print(f"  {C.BOLD}Experiment:{C.RESET} {args.exp}")
    print(f"  {C.BOLD}Samples:{C.RESET}    {len(entries)}")
    print(f"  {C.BOLD}Score range:{C.RESET} [{C.RED}{min(scores):.4f}{C.RESET}, "
          f"{C.GREEN}{max(scores):.4f}{C.RESET}]")
    print(f"  {C.BOLD}Score mean:{C.RESET}  {sum(scores)/len(scores):.4f}")
    print(f"  {C.BOLD}Score median:{C.RESET} {scores[len(scores)//2]:.4f}")

    # ── Score histogram ──
    print()
    print(f"  {C.BOLD}Score Distribution:{C.RESET}")
    n_bins = 10
    lo, hi = min(scores), max(scores)
    if hi == lo:
        hi = lo + 1
    bin_width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for s in scores:
        idx = min(int((s - lo) / bin_width), n_bins - 1)
        bins[idx] += 1
    max_count = max(bins) or 1
    hist_width = 40
    for i, count in enumerate(bins):
        edge = lo + i * bin_width
        blen = int(count / max_count * hist_width)
        print(f"    {edge:6.2f} │{'█' * blen}{' ' * (hist_width - blen)}│ {count}")

    if args.stats:
        print()
        sys.exit(0)

    # ── Show entries ──
    if args.all:
        # Compact table view for --all
        print()
        print(f"  {C.BOLD}{'#':>4}  {'Score':>8}  {'TurnR':>30}  Behavior{C.RESET}")
        print(f"  {C.DIM}{'─'*4}  {'─'*8}  {'─'*30}  {'─'*50}{C.RESET}")
        for i, e in enumerate(entries):
            s = e["score"]
            behavior = extract_behavior(e["input"])[:50]
            tr = e.get("turn_rewards", [])
            tr_str = " ".join(f"{r:.2f}" for r in tr)
            # Color by score
            if s >= 4.0:
                sc = C.GREEN
            elif s >= 2.0:
                sc = C.YELLOW
            else:
                sc = C.RED
            rank = i + 1
            print(f"  {rank:>4}  {sc}{s:>8.4f}{C.RESET}  {C.DIM}{tr_str:>30}{C.RESET}  {behavior}")
        print()

        # Prompt for detail view
        print(f"{C.DIM}  Tip: use 'python view_rollout.py {step} --top N' to see detailed top/bottom N{C.RESET}")
    else:
        n = min(args.top, len(entries))

        # Bottom-N (worst)
        for i in range(n):
            label = f"▼ WORST #{i+1}  (rank {i+1}/{len(entries)})"
            print_entry(entries[i], label, C.RED)

        if n < len(entries):
            print(f"\n{C.DIM}  ... {len(entries) - 2*n} samples omitted ...{C.RESET}\n")

        # Top-N (best)
        for i in range(n):
            idx = len(entries) - n + i
            label = f"▲ BEST #{n-i}  (rank {idx+1}/{len(entries)})"
            print_entry(entries[idx], label, C.GREEN)

    print()


if __name__ == "__main__":
    main()
