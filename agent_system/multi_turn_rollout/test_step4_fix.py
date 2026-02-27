"""
Verification test for Step 4 (Turn -> Token mapping) bug fix.

Uses actual data from:
  examples/multiturn_grpo_trainer/run_logs/multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl

The bug: Each step entry's response tokens start at pos 0, but turn_token_mask
is the full trajectory mask which always starts with turn 0. So step 1+ entries
got turn 0's advantage instead of the correct turn's advantage.

The fix: Use step_index to directly look up the correct turn's advantage and
assign it uniformly to all response tokens of that step entry.
"""

import torch
import numpy as np
import json
import os

from verl.trainer.ppo.core_algos import compute_multiturn_grpo_advantage


COLORS = {
    0: "\033[94m",   # blue
    1: "\033[92m",   # green
    2: "\033[93m",   # yellow
    3: "\033[91m",   # red
    4: "\033[95m",   # magenta
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"


def color_for_turn(turn):
    return COLORS.get(turn, "\033[96m")


def load_jsonl_trajectory():
    """Load the first trajectory from the JSONL log file."""
    jsonl_path = os.path.join(
        os.path.dirname(__file__),
        "../../examples/multiturn_grpo_trainer/run_logs/"
        "multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl",
    )
    jsonl_path = os.path.abspath(jsonl_path)

    with open(jsonl_path) as f:
        lines = [json.loads(line) for line in f]

    # First trajectory: lines 0,1,2,3,20 (identified by matching turn_rewards)
    # Sort by number of assistant turns in input to get step order
    first_traj_lines = [lines[i] for i in [20, 0, 1, 2, 3]]
    # line 20: 1 assistant turn -> step 0
    # line 0:  2 assistant turns -> step 1
    # line 1:  3 assistant turns -> step 2
    # line 2:  4 assistant turns -> step 3
    # line 3:  5 assistant turns -> step 4

    return first_traj_lines


def visualize_token_advantage_map(
    step, n_tokens, adv_fixed, adv_buggy, max_tokens, A_tilde, output_text
):
    """Visualize how advantages are assigned to tokens for one step entry."""
    c = color_for_turn(step)

    # Show output text preview
    preview = output_text[:100].replace("\n", "\\n")
    print(f"  {c}{BOLD}output:{RESET} \"{preview}...\"")
    print()

    # Token-level advantage bar
    bar_width = min(max_tokens, 60)
    scale = bar_width / max_tokens

    # Fixed advantages
    fixed_vals = adv_fixed[step].tolist()
    buggy_vals = adv_buggy[step].tolist()

    # Build visual bars
    # Fixed bar
    fixed_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos / scale)
        if real_pos < n_tokens:
            fixed_bar += f"{c}#{RESET}"
        else:
            fixed_bar += f"{DIM}.{RESET}"

    # Buggy bar - color by what turn the buggy code actually assigns
    # Buggy code reads turn_token_mask from pos 0, so it picks up turn 0 tokens
    buggy_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos / scale)
        if real_pos < n_tokens:
            # Buggy always assigns turn 0's advantage to the first tokens
            buggy_c = color_for_turn(0)
            buggy_bar += f"{buggy_c}#{RESET}"
        else:
            buggy_bar += f"{DIM}.{RESET}"

    expected_adv = A_tilde[step]
    fixed_adv = fixed_vals[0] if n_tokens > 0 else 0.0
    buggy_adv = buggy_vals[0] if n_tokens > 0 else 0.0

    print(f"  {BOLD}Fixed:{RESET}  [{fixed_bar}] adv={fixed_adv:+.4f} (turn {step})")
    print(f"  {BOLD}Buggy:{RESET}  [{buggy_bar}] adv={buggy_adv:+.4f} (turn 0)")

    match = abs(fixed_adv - expected_adv) < 1e-6
    buggy_match = abs(buggy_adv - expected_adv) < 1e-6
    fixed_tag = f"{BG_GREEN} CORRECT {RESET}" if match else f"{BG_RED} WRONG {RESET}"
    buggy_tag = f"{BG_GREEN} CORRECT {RESET}" if buggy_match else f"{BG_RED} WRONG {RESET}"
    print(f"          fixed: {fixed_tag}  expected={expected_adv:+.4f}")
    print(f"          buggy: {buggy_tag}  expected={expected_adv:+.4f}")


def test_step4_fix_with_jsonl_data():
    """
    Simulate the real batch structure and verify correct advantage assignment.
    Uses actual input/output from the JSONL log to visually show mask + advantage.
    """
    traj_data = load_jsonl_trajectory()

    # From the JSONL data (first trajectory)
    turn_rewards_from_data = traj_data[0]["turn_rewards"]
    turn_advantages_from_data = traj_data[0]["turn_advantages"]

    num_turns = 5
    # Actual token counts from Qwen2.5-7B tokenizer on the output text
    tokens_per_step = [745, 822, 842, 742, 406]
    max_response_len = max(tokens_per_step)

    # All entries share the same trajectory data
    turn_rewards = [turn_rewards_from_data] * num_turns
    turn_texts = [["text"] * num_turns] * num_turns

    # Build turn_token_mask: full trajectory mask accumulated across all steps
    full_mask = []
    for t, n_tok in enumerate(tokens_per_step):
        full_mask.extend([t] * n_tok)
    turn_token_mask = [full_mask] * num_turns

    # response_mask: each entry only has its own step's tokens
    response_mask = torch.zeros((num_turns, max_response_len), dtype=torch.float32)
    for i, n_tok in enumerate(tokens_per_step):
        response_mask[i, :n_tok] = 1.0

    index = np.array(["group_0"] * num_turns, dtype=object)
    traj_index = np.array(["traj_0"] * num_turns, dtype=object)
    step_index = list(range(num_turns))

    # ── Run FIXED ─────────────────────────────────────────────────────
    adv_fixed, _, A_tilde_fixed = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        step_index=step_index,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    # ── Run BUGGY (old behavior) ──────────────────────────────────────
    adv_buggy, _, A_tilde_buggy = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        step_index=None,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    # ══════════════════════════════════════════════════════════════════
    # 1. Trajectory Overview
    # ══════════════════════════════════════════════════════════════════
    print()
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  TRAJECTORY OVERVIEW (from 1.jsonl, first trajectory){RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    # Turn color legend
    print(f"  Turn colors: ", end="")
    for t in range(num_turns):
        c = color_for_turn(t)
        print(f"{c}[Turn {t}]{RESET}  ", end="")
    print()
    print()

    # Rewards and A_tilde table
    print(f"  {'Turn':>5} | {'Reward':>8} | {'Normalized A':>13} | {'A_tilde (gamma-discounted)':>27} | {'Tokens':>7}")
    print(f"  {'-' * 72}")
    for t in range(num_turns):
        c = color_for_turn(t)
        print(
            f"  {c}{t:>5}{RESET} | "
            f"{turn_rewards_from_data[t]:>8.4f} | "
            f"{A_tilde_fixed[0][t] / (0.9 ** 0 if t == 0 else 1):>13.4f} | "
            f"{c}{A_tilde_fixed[0][t]:>27.4f}{RESET} | "
            f"{tokens_per_step[t]:>7}"
        )
    print()

    # ══════════════════════════════════════════════════════════════════
    # 2. Full trajectory token mask visualization
    # ══════════════════════════════════════════════════════════════════
    total_tokens = sum(tokens_per_step)
    bar_width = 60
    print(f"  {BOLD}Full trajectory turn_token_mask{RESET} ({total_tokens} tokens total):")
    print(f"  [", end="")
    for pos in range(bar_width):
        real_pos = int(pos * total_tokens / bar_width)
        # Find which turn this position belongs to
        cumsum = 0
        for t, n_tok in enumerate(tokens_per_step):
            cumsum += n_tok
            if real_pos < cumsum:
                c = color_for_turn(t)
                print(f"{c}{'=' * 1}{RESET}", end="")
                break
    print("]")

    # Ticks
    print(f"  ", end="")
    cumsum = 0
    for t, n_tok in enumerate(tokens_per_step):
        mid_pos = cumsum + n_tok // 2
        bar_pos = int(mid_pos * bar_width / total_tokens)
        print(f"\033[{bar_pos + 3}G{color_for_turn(t)}T{t}{RESET}", end="")
        cumsum += n_tok
    print()
    print()

    # ══════════════════════════════════════════════════════════════════
    # 3. Per-step entry: input/output + mask + advantage visualization
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  PER-STEP BATCH ENTRIES: Mask & Advantage Assignment{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    all_fixed_ok = True
    any_buggy_wrong = False

    for step in range(num_turns):
        c = color_for_turn(step)
        n_tok = tokens_per_step[step]

        print(f"  {c}{BOLD}--- Step {step} (batch entry) ---{RESET}")
        print(f"  {BOLD}input:{RESET}  ...conversation with {step + 1} assistant turn(s) "
              f"({len(traj_data[step]['input'])} chars)")

        visualize_token_advantage_map(
            step, n_tok, adv_fixed, adv_buggy,
            max_response_len, A_tilde_fixed[step], traj_data[step]["output"]
        )

        expected = A_tilde_fixed[step][step]
        fixed_val = adv_fixed[step, 0].item()
        buggy_val = adv_buggy[step, 0].item()

        fixed_ok = abs(fixed_val - expected) < 1e-6
        buggy_ok = abs(buggy_val - expected) < 1e-6
        if not fixed_ok:
            all_fixed_ok = False
        if not buggy_ok:
            any_buggy_wrong = True

        # Verify uniformity
        active_advs = adv_fixed[step, :n_tok]
        pad_advs = adv_fixed[step, n_tok:]
        is_uniform = torch.all(active_advs == active_advs[0]).item()
        is_pad_zero = torch.all(pad_advs == 0).item()

        print(f"  {BOLD}uniform across {n_tok} tokens:{RESET} {is_uniform}  "
              f"{BOLD}padding zero:{RESET} {is_pad_zero}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # 4. Side-by-side comparison table
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  COMPARISON TABLE: Fixed vs Buggy{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()
    header = (
        f"  {'Step':>5} | {'Tokens':>6} | "
        f"{'Expected':>10} | {'Fixed':>10} | {'Buggy':>10} | "
        f"{'Fixed':>7} | {'Buggy':>7}"
    )
    print(header)
    print(f"  {'-' * 72}")

    for step in range(num_turns):
        c = color_for_turn(step)
        n_tok = tokens_per_step[step]
        expected = A_tilde_fixed[step][step]
        fixed_val = adv_fixed[step, 0].item()
        buggy_val = adv_buggy[step, 0].item()

        fixed_ok = abs(fixed_val - expected) < 1e-6
        buggy_ok = abs(buggy_val - expected) < 1e-6

        fixed_mark = f"{BG_GREEN} PASS {RESET}" if fixed_ok else f"{BG_RED} FAIL {RESET}"
        buggy_mark = f"{BG_GREEN} PASS {RESET}" if buggy_ok else f"{BG_RED} FAIL {RESET}"

        print(
            f"  {c}{step:>5}{RESET} | {n_tok:>6} | "
            f"{expected:>+10.4f} | {fixed_val:>+10.4f} | {buggy_val:>+10.4f} | "
            f"{fixed_mark} | {buggy_mark}"
        )

    print()

    # ══════════════════════════════════════════════════════════════════
    # 5. Bug explanation diagram
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  BUG EXPLANATION: Why step 1+ entries get wrong advantage{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()
    print(f"  Each step entry's 'responses' tensor starts at pos 0.")
    print(f"  But turn_token_mask is the FULL trajectory mask:")
    print()

    # Show what buggy code reads for each step
    for step in range(num_turns):
        c = color_for_turn(step)
        n_tok = tokens_per_step[step]

        # What the buggy code reads: turn_token_mask[0:n_tok]
        buggy_slice = full_mask[:n_tok]
        turns_in_slice = sorted(set(buggy_slice))
        counts = {t: buggy_slice.count(t) for t in turns_in_slice}

        print(f"  {c}Step {step}{RESET} ({n_tok} response tokens):")
        print(f"    Buggy reads turn_token_mask[0:{n_tok}] -> turns {turns_in_slice}")
        for t in turns_in_slice:
            tc = color_for_turn(t)
            wrong = " <-- WRONG!" if t != step else ""
            print(f"      {tc}turn {t}{RESET}: {counts[t]} tokens get A_tilde[{t}]={A_tilde_fixed[step][t]:+.4f}{wrong}")

        print(f"    {BOLD}Fixed uses step_index={step}{RESET} -> "
              f"ALL {n_tok} tokens get {c}A_tilde[{step}]={A_tilde_fixed[step][step]:+.4f}{RESET}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # 6. Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"  Fixed code : all steps correct = {all_fixed_ok}")
    print(f"  Buggy code : has wrong assignments = {any_buggy_wrong}")
    print()

    assert all_fixed_ok, "Fixed code should assign correct advantages to all steps"
    assert any_buggy_wrong, "Buggy code should have wrong assignments for step 1+"

    print(f"  {BG_GREEN}{BOLD} ALL ASSERTIONS PASSED {RESET}")
    print()


def test_step4_fix_multi_trajectory():
    """
    Test with 2 trajectories (2 groups x 1 traj each) to verify group
    normalization + step_index interaction.
    """
    turn_rewards_traj0 = [0.5, 0.8, 1.0]
    turn_rewards_traj1 = [0.2, 0.3, 0.1]

    turn_rewards = [
        turn_rewards_traj0, turn_rewards_traj0, turn_rewards_traj0,
        turn_rewards_traj1, turn_rewards_traj1, turn_rewards_traj1,
    ]
    turn_texts = [["a", "b", "c"]] * 6
    turn_token_mask = [
        [0, 0, 1, 1, 2], [0, 0, 1, 1, 2], [0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 2, 2],
    ]

    tokens_per_step = [20, 25, 15, 30, 20, 25]
    max_len = max(tokens_per_step)
    response_mask = torch.zeros((6, max_len), dtype=torch.float32)
    for i, n in enumerate(tokens_per_step):
        response_mask[i, :n] = 1.0

    index = np.array(["g0", "g0", "g0", "g1", "g1", "g1"], dtype=object)
    traj_index = np.array(["t0", "t0", "t0", "t1", "t1", "t1"], dtype=object)
    step_index = [0, 1, 2, 0, 1, 2]

    adv, _, A_tilde = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        step_index=step_index,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  Multi-trajectory test (2 groups x 1 traj x 3 turns){RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    for i in range(6):
        traj = 0 if i < 3 else 1
        step = i % 3
        n_tok = tokens_per_step[i]
        c = color_for_turn(step)
        expected_adv = A_tilde[i][step]
        actual_adv = adv[i, 0].item()
        ok = abs(actual_adv - expected_adv) < 1e-6

        mark = f"{BG_GREEN} PASS {RESET}" if ok else f"{BG_RED} FAIL {RESET}"
        print(
            f"  Entry {i} (traj={traj}, {c}step={step}{RESET}): "
            f"expected={expected_adv:+.6f}, actual={actual_adv:+.6f}  {mark}"
        )
        assert ok, f"Entry {i} got wrong advantage"
        assert torch.all(adv[i, :n_tok] == adv[i, 0]).item(), f"Entry {i} not uniform"
        assert torch.all(adv[i, n_tok:] == 0).item(), f"Entry {i} padding not zero"

    print()
    print(f"  {BG_GREEN}{BOLD} ALL ASSERTIONS PASSED {RESET}")
    print()


if __name__ == "__main__":
    test_step4_fix_with_jsonl_data()
    test_step4_fix_multi_trajectory()
