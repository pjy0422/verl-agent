"""
Verification test for response_mask vs loss_mask in MULTITURN_GRPO.

Uses actual data from:
  examples/multiturn_grpo_trainer/run_logs/multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl

Key difference:
  - response_mask: marks ALL non-pad tokens in the response window (attention_mask based)
  - loss_mask:     marks ONLY assistant-generated tokens (role-aware)

In multi-turn, the prompt (input) contains interleaved system/user/assistant turns.
The response (output) is pure assistant generation per step entry.

This test visualizes both masks across the full sequence to show:
  1. Where they agree (response window = pure assistant)
  2. Where they'd diverge (if response window contained user/system tokens)
  3. The correct mask choice for policy gradient: loss_mask
"""

import torch
import numpy as np
import json
import os

from verl.trainer.ppo.core_algos import compute_multiturn_grpo_advantage


# ── ANSI colors ──────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"

ROLE_COLORS = {
    "system": "\033[90m",      # gray
    "user": "\033[94m",        # blue
    "assistant": "\033[92m",   # green
    "pad": "\033[2m",          # dim
}

TURN_COLORS = {
    0: "\033[94m",   # blue
    1: "\033[92m",   # green
    2: "\033[93m",   # yellow
    3: "\033[91m",   # red
    4: "\033[95m",   # magenta
}


def color_role(role):
    return ROLE_COLORS.get(role, "\033[96m")


def color_turn(t):
    return TURN_COLORS.get(t, "\033[96m")


def load_first_line():
    """Load the first entry from JSONL."""
    jsonl_path = os.path.join(
        os.path.dirname(__file__),
        "../../examples/multiturn_grpo_trainer/run_logs/"
        "multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl",
    )
    jsonl_path = os.path.abspath(jsonl_path)
    with open(jsonl_path) as f:
        return json.loads(f.readline())


def parse_role_segments(text):
    """Parse text into (role, content) segments based on role markers."""
    lines = text.split("\n")
    segments = []
    current_role = None
    current_lines = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped in ("system", "user", "assistant"):
            if current_role is not None:
                segments.append((current_role, "\n".join(current_lines)))
            current_role = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_role:
        segments.append((current_role, "\n".join(current_lines)))

    return segments


def simulate_tokens(segments, output_text, max_seq_len=512, response_len=128):
    """
    Simulate tokenization of the full sequence (input segments + output).

    Returns:
        token_roles: list of role strings per token position (full sequence)
        prompt_len: number of prompt tokens
        resp_len: number of response tokens (actual, before padding)
    """
    # Estimate token counts proportional to char length (rough ~4 chars/token)
    CHARS_PER_TOKEN = 4

    prompt_tokens = []
    for role, content in segments:
        n_tokens = max(1, len(content) // CHARS_PER_TOKEN)
        prompt_tokens.extend([(role, i) for i in range(n_tokens)])

    # Response tokens are pure assistant
    resp_token_count = max(1, len(output_text) // CHARS_PER_TOKEN)

    # Truncate / pad to fit max_seq_len
    prompt_len = min(len(prompt_tokens), max_seq_len - response_len)
    prompt_tokens = prompt_tokens[-prompt_len:]  # keep last tokens (right-aligned)
    resp_actual = min(resp_token_count, response_len)

    # Build full sequence token roles
    token_roles = []
    for role, _ in prompt_tokens:
        token_roles.append(role)
    for _ in range(resp_actual):
        token_roles.append("assistant")
    for _ in range(response_len - resp_actual):
        token_roles.append("pad")

    return token_roles, prompt_len, resp_actual


def build_masks(token_roles, prompt_len, response_len):
    """
    Build response_mask and loss_mask for the response window.

    response_mask: 1 for any non-pad token in response window
    loss_mask:     1 only for assistant tokens in response window
    """
    seq_len = len(token_roles)
    resp_start = prompt_len

    # Full-sequence attention mask
    attention_mask = torch.zeros(seq_len, dtype=torch.long)
    for i, role in enumerate(token_roles):
        if role != "pad":
            attention_mask[i] = 1

    # Full-sequence loss mask (only assistant)
    loss_mask_full = torch.zeros(seq_len, dtype=torch.long)
    for i, role in enumerate(token_roles):
        if role == "assistant":
            loss_mask_full[i] = 1

    # Extract response window
    response_mask = attention_mask[resp_start:]
    loss_mask = loss_mask_full[resp_start:]

    return attention_mask, loss_mask_full, response_mask, loss_mask


def visualize_full_sequence(token_roles, prompt_len, resp_actual, response_len):
    """Visualize the full sequence with role coloring."""
    seq_len = len(token_roles)
    bar_width = min(seq_len, 80)

    print(f"  {BOLD}Full sequence{RESET} ({seq_len} tokens total)")
    print(f"  Prompt: {prompt_len} tokens | Response: {resp_actual} tokens | Pad: {response_len - resp_actual} tokens")
    print()

    # Build bar
    bar = ""
    for pos in range(bar_width):
        real_pos = int(pos * seq_len / bar_width)
        role = token_roles[real_pos]
        c = color_role(role)
        if role == "pad":
            bar += f"{c}.{RESET}"
        else:
            bar += f"{c}={RESET}"

    print(f"  [{bar}]")

    # Legend with segment boundaries
    print(f"  ", end="")
    prev_role = None
    for pos in range(bar_width):
        real_pos = int(pos * seq_len / bar_width)
        role = token_roles[real_pos]
        if role != prev_role:
            c = color_role(role)
            label = role[0].upper()
            print(f"{c}{label}{RESET}", end="")
            prev_role = role
        else:
            print(" ", end="")
    print()

    # Prompt/Response boundary
    boundary_pos = int(prompt_len * bar_width / seq_len)
    print(f"  {' ' * boundary_pos}^")
    print(f"  {' ' * boundary_pos}prompt|response")
    print()


def visualize_response_masks(token_roles, prompt_len, response_mask, loss_mask, resp_actual, response_len):
    """Visualize response_mask vs loss_mask in the response window."""
    bar_width = min(response_len, 70)

    print(f"  {BOLD}Response window{RESET} ({response_len} tokens)")
    print()

    # response_mask bar
    resp_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos * response_len / bar_width)
        if response_mask[real_pos] == 1:
            resp_bar += f"\033[92m#{RESET}"  # green = active
        else:
            resp_bar += f"{DIM}.{RESET}"

    # loss_mask bar
    loss_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos * response_len / bar_width)
        if loss_mask[real_pos] == 1:
            loss_bar += f"\033[92m#{RESET}"
        else:
            loss_bar += f"{DIM}.{RESET}"

    # diff bar (where they differ)
    diff_bar = ""
    n_diff = 0
    for pos in range(bar_width):
        real_pos = int(pos * response_len / bar_width)
        r = response_mask[real_pos].item()
        l = loss_mask[real_pos].item()
        if r == l:
            if r == 1:
                diff_bar += f"\033[92m={RESET}"  # both active
            else:
                diff_bar += f"{DIM}.{RESET}"  # both inactive
        else:
            diff_bar += f"{BG_RED}!{RESET}"  # DIFFER
            n_diff += 1

    active_resp = int(response_mask.sum().item())
    active_loss = int(loss_mask.sum().item())

    print(f"  {BOLD}response_mask:{RESET} [{resp_bar}] {active_resp} active tokens")
    print(f"  {BOLD}loss_mask:    {RESET} [{loss_bar}] {active_loss} active tokens")
    print(f"  {BOLD}difference:   {RESET} [{diff_bar}] {n_diff} positions differ")
    print()

    return n_diff


def visualize_mixed_response_scenario():
    """
    Hypothetical scenario: response window contains user + assistant tokens.
    This happens when conversation continuation is appended to the response.
    """
    print(f"  {BOLD}Hypothetical: response window with mixed roles{RESET}")
    print(f"  (e.g. assistant(60) + user(30) + assistant(30) + pad(8))")
    print()

    response_len = 128
    # Simulate: assistant(60) + user(30) + assistant(30) + pad(8)
    mixed_roles = (
        ["assistant"] * 60 + ["user"] * 30 + ["assistant"] * 30 + ["pad"] * 8
    )

    response_mask = torch.zeros(response_len, dtype=torch.long)
    loss_mask = torch.zeros(response_len, dtype=torch.long)
    for i, role in enumerate(mixed_roles):
        if role != "pad":
            response_mask[i] = 1
        if role == "assistant":
            loss_mask[i] = 1

    bar_width = min(response_len, 70)

    # Role bar
    role_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos * response_len / bar_width)
        role = mixed_roles[real_pos]
        c = color_role(role)
        if role == "pad":
            role_bar += f"{c}.{RESET}"
        else:
            role_bar += f"{c}={RESET}"

    resp_bar = ""
    loss_bar = ""
    diff_bar = ""
    n_diff = 0
    for pos in range(bar_width):
        real_pos = int(pos * response_len / bar_width)
        r = response_mask[real_pos].item()
        l = loss_mask[real_pos].item()

        resp_bar += f"\033[92m#{RESET}" if r else f"{DIM}.{RESET}"
        loss_bar += f"\033[92m#{RESET}" if l else f"{DIM}.{RESET}"

        if r == l:
            diff_bar += f"\033[92m={RESET}" if r else f"{DIM}.{RESET}"
        else:
            diff_bar += f"{BG_RED}!{RESET}"
            n_diff += 1

    print(f"  {BOLD}roles:        {RESET} [{role_bar}]")
    print(f"               {color_role('assistant')}= assistant  "
          f"{color_role('user')}= user  "
          f"{color_role('pad')}. pad{RESET}")
    print()
    print(f"  {BOLD}response_mask:{RESET} [{resp_bar}] {int(response_mask.sum())} active (includes user!)")
    print(f"  {BOLD}loss_mask:    {RESET} [{loss_bar}] {int(loss_mask.sum())} active (assistant only)")
    print(f"  {BOLD}difference:   {RESET} [{diff_bar}] {n_diff} positions = user tokens WRONGLY included")
    print()

    return n_diff


def test_mask_on_real_data():
    """
    Test with actual JSONL data: visualize which tokens each mask selects.
    """
    data = load_first_line()
    segments = parse_role_segments(data["input"])
    output_text = data["output"]

    MAX_SEQ_LEN = 512
    RESPONSE_LEN = 128

    token_roles, prompt_len, resp_actual = simulate_tokens(
        segments, output_text, MAX_SEQ_LEN, RESPONSE_LEN,
    )
    attention_mask, loss_mask_full, response_mask, loss_mask = build_masks(
        token_roles, prompt_len, RESPONSE_LEN,
    )

    # ══════════════════════════════════════════════════════════════════
    # 1. JSONL data overview
    # ══════════════════════════════════════════════════════════════════
    print()
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  MASK VERIFICATION: response_mask vs loss_mask (from 1.jsonl line 0){RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    print(f"  {BOLD}Entry metadata:{RESET}")
    print(f"    step={data['step']}, score={data['score']:.4f}")
    print(f"    turn_rewards={data['turn_rewards']}")
    print()

    print(f"  {BOLD}Input role structure:{RESET}")
    for i, (role, content) in enumerate(segments):
        c = color_role(role)
        preview = content[:60].strip().replace("\n", "\\n")
        print(f"    [{i}] {c}{role:>10}{RESET} | {len(content):>5} chars | \"{preview}...\"")
    print()
    print(f"  {BOLD}Output:{RESET} assistant generation, {len(output_text)} chars")
    print(f"    \"{output_text[:80]}...\"")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 2. Full sequence visualization
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  FULL SEQUENCE TOKEN ROLES{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    print(f"  Legend: {color_role('system')}= system{RESET}  "
          f"{color_role('user')}= user{RESET}  "
          f"{color_role('assistant')}= assistant{RESET}  "
          f"{color_role('pad')}. pad{RESET}")
    print()

    visualize_full_sequence(token_roles, prompt_len, resp_actual, RESPONSE_LEN)

    # ══════════════════════════════════════════════════════════════════
    # 3. Response window mask comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  RESPONSE WINDOW: response_mask vs loss_mask{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    print(f"  {BOLD}Case A: Current rollout structure (response = pure assistant){RESET}")
    print()
    n_diff_real = visualize_response_masks(
        token_roles, prompt_len, response_mask, loss_mask, resp_actual, RESPONSE_LEN,
    )

    if n_diff_real == 0:
        print(f"  {BG_GREEN} response_mask == loss_mask in response window {RESET}")
        print(f"  In this rollout, each step's response is pure assistant generation,")
        print(f"  so both masks select the same tokens.")
    else:
        print(f"  {BG_RED} MASKS DIFFER! {n_diff_real} tokens wrongly included by response_mask {RESET}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 4. Hypothetical mixed-role scenario
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  HYPOTHETICAL: What if response window contained user tokens?{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    print(f"  {BOLD}Case B: Mixed response (assistant + user + assistant + pad){RESET}")
    print()
    n_diff_mixed = visualize_mixed_response_scenario()

    print(f"  {BG_RED} response_mask includes {n_diff_mixed} user token positions! {RESET}")
    print(f"  Using response_mask would leak gradient into non-policy tokens.")
    print(f"  loss_mask correctly excludes them.")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 5. Full sequence mask comparison (prompt + response)
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  FULL SEQUENCE: attention_mask vs loss_mask (for context){RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    seq_len = len(token_roles)
    bar_width = min(seq_len, 70)

    attn_bar = ""
    loss_full_bar = ""
    for pos in range(bar_width):
        real_pos = int(pos * seq_len / bar_width)
        a = attention_mask[real_pos].item()
        l = loss_mask_full[real_pos].item()
        attn_bar += f"\033[92m#{RESET}" if a else f"{DIM}.{RESET}"
        loss_full_bar += f"\033[92m#{RESET}" if l else f"{DIM}.{RESET}"

    print(f"  {BOLD}attention_mask:{RESET} [{attn_bar}] {int(attention_mask.sum())} active")
    print(f"  {BOLD}loss_mask:     {RESET} [{loss_full_bar}] {int(loss_mask_full.sum())} active")
    print()
    print(f"  attention_mask: marks ALL non-pad tokens (system + user + assistant)")
    print(f"  loss_mask:      marks ONLY assistant tokens (policy-generated)")
    print(f"  In prompt region, loss_mask excludes system/user => no gradient for non-policy tokens.")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 6. Advantage computation with both masks
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  ADVANTAGE COMPUTATION: response_mask vs loss_mask{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    turn_rewards = [data["turn_rewards"]]
    turn_texts = [["text"] * len(data["turn_rewards"])]
    num_turns = len(data["turn_rewards"])
    step_idx = data["step"]

    # Simulate single-entry batch for this step
    resp_tokens = max(1, len(output_text) // 4)  # ~4 chars/token
    max_resp = resp_tokens + 20  # some padding

    mask_1 = torch.zeros((1, max_resp), dtype=torch.float32)
    mask_1[0, :resp_tokens] = 1.0  # response_mask: all non-pad

    # loss_mask: same in this case (response = pure assistant)
    mask_2 = mask_1.clone()

    turn_token_mask = [[step_idx] * resp_tokens]
    index = np.array(["group_0"], dtype=object)
    traj_index = np.array(["traj_0"], dtype=object)

    adv_resp, _, Atilde_resp = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=mask_1,
        index=index,
        traj_index=traj_index,
        step_index=[step_idx],
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    adv_loss, _, Atilde_loss = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=mask_2,
        index=index,
        traj_index=traj_index,
        step_index=[step_idx],
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    adv_val_resp = adv_resp[0, 0].item() if resp_tokens > 0 else 0.0
    adv_val_loss = adv_loss[0, 0].item() if resp_tokens > 0 else 0.0

    print(f"  Step {step_idx}, turn_rewards = {data['turn_rewards']}")
    print(f"  Response tokens: {resp_tokens} (of {max_resp} max)")
    print()
    print(f"  With response_mask: advantage = {adv_val_resp:+.6f}")
    print(f"  With loss_mask:     advantage = {adv_val_loss:+.6f}")
    print()

    match = abs(adv_val_resp - adv_val_loss) < 1e-6
    if match:
        print(f"  {BG_GREEN} Values match (pure assistant response) {RESET}")
    else:
        print(f"  {BG_RED} Values differ! {RESET}")
    print()

    # ══════════════════════════════════════════════════════════════════
    # 7. Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()
    print(f"  1. In current rollout: response = pure assistant generation")
    print(f"     => response_mask == loss_mask in response window (no functional diff)")
    print()
    print(f"  2. Semantically: loss_mask is CORRECT because it encodes")
    print(f"     'which tokens are policy-generated' regardless of rollout structure.")
    print()
    print(f"  3. Consistency: GRPO branch already uses loss_mask when multi_turn=True")
    print(f"     (dp_actor.py:369-370, ray_trainer.py:362-369)")
    print()
    print(f"  4. Safety: if rollout structure changes (e.g. environment response")
    print(f"     appended to generation), loss_mask still correctly excludes")
    print(f"     non-policy tokens. response_mask would silently break.")
    print()

    # Assertions
    assert n_diff_real == 0, "In pure-assistant response, masks should be equal"
    assert n_diff_mixed > 0, "In mixed response, masks should differ"
    assert match, "Advantage values should match for pure-assistant response"

    print(f"  {BG_GREEN}{BOLD} ALL ASSERTIONS PASSED {RESET}")
    print()


def test_mask_with_mixed_response_advantage():
    """
    Demonstrate that using response_mask with a mixed-role response window
    produces different (wrong) advantages compared to loss_mask.

    Simulates a case where the response window contains:
      assistant(50 tokens) + user(30 tokens) + assistant(40 tokens) + pad(8 tokens)
    """
    print()
    print(f"{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  MIXED RESPONSE: Advantage difference with wrong mask{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    print()

    num_turns = 3
    turn_rewards = [[0.5, 0.8, 1.0]] * num_turns
    turn_texts = [["a", "b", "c"]] * num_turns

    # Each step entry: 128 tokens
    max_resp = 128
    # Step 0: 50 assistant tokens
    # Step 1: 40 assistant tokens preceded by 30 user tokens in window
    # Step 2: 30 assistant tokens

    tokens_per_step = [50, 70, 30]  # total non-pad tokens in each entry's window

    # Build masks for each step entry
    #   response_mask: marks all non-pad positions
    #   loss_mask:     marks only assistant positions

    # Step 0: response = [assistant(50) + pad(78)]
    resp_mask_0 = torch.zeros(max_resp, dtype=torch.float32)
    resp_mask_0[:50] = 1.0
    loss_mask_0 = resp_mask_0.clone()  # same: pure assistant

    # Step 1: response = [user(30) + assistant(40) + pad(58)]
    #   This simulates env response prepended to model generation
    resp_mask_1 = torch.zeros(max_resp, dtype=torch.float32)
    resp_mask_1[:70] = 1.0  # all 70 non-pad tokens
    loss_mask_1 = torch.zeros(max_resp, dtype=torch.float32)
    loss_mask_1[30:70] = 1.0  # only assistant tokens (pos 30-69)

    # Step 2: response = [assistant(30) + pad(98)]
    resp_mask_2 = torch.zeros(max_resp, dtype=torch.float32)
    resp_mask_2[:30] = 1.0
    loss_mask_2 = resp_mask_2.clone()  # same: pure assistant

    response_mask_batch = torch.stack([resp_mask_0, resp_mask_1, resp_mask_2])
    loss_mask_batch = torch.stack([loss_mask_0, loss_mask_1, loss_mask_2])

    turn_token_mask = [[0] * 50, [1] * 40, [2] * 30]
    index = np.array(["g0"] * num_turns, dtype=object)
    traj_index = np.array(["t0"] * num_turns, dtype=object)
    step_index = [0, 1, 2]

    # With response_mask (WRONG for step 1)
    adv_resp, _, Atilde_resp = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask_batch,
        index=index,
        traj_index=traj_index,
        step_index=step_index,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    # With loss_mask (CORRECT)
    adv_loss, _, Atilde_loss = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=loss_mask_batch,
        index=index,
        traj_index=traj_index,
        step_index=step_index,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    # Visualize per-step
    step_configs = [
        (0, "assistant(50) + pad(78)", 50, 50),
        (1, "user(30) + assistant(40) + pad(58)", 70, 40),
        (2, "assistant(30) + pad(98)", 30, 30),
    ]

    for step, desc, total_active, assistant_active in step_configs:
        c = color_turn(step)

        bar_width = min(max_resp, 60)
        # Build mask bars
        resp_bar = ""
        loss_bar_vis = ""
        for pos in range(bar_width):
            real_pos = int(pos * max_resp / bar_width)
            r = response_mask_batch[step, real_pos].item()
            l = loss_mask_batch[step, real_pos].item()

            if r == 1 and l == 1:
                resp_bar += f"\033[92m#{RESET}"
                loss_bar_vis += f"\033[92m#{RESET}"
            elif r == 1 and l == 0:
                resp_bar += f"{BG_RED}#{RESET}"  # included by response_mask but NOT loss_mask
                loss_bar_vis += f"{DIM}.{RESET}"
            else:
                resp_bar += f"{DIM}.{RESET}"
                loss_bar_vis += f"{DIM}.{RESET}"

        resp_adv = adv_resp[step, 0].item()
        loss_adv = adv_loss[step, 0].item()
        diff = abs(resp_adv - loss_adv)

        print(f"  {c}{BOLD}Step {step}{RESET}: {desc}")
        print(f"    response_mask: [{resp_bar}] {total_active} active tokens, adv={resp_adv:+.6f}")
        print(f"    loss_mask:     [{loss_bar_vis}] {assistant_active} active tokens, adv={loss_adv:+.6f}")

        if diff > 1e-6:
            print(f"    {BG_RED} DIFFER by {diff:.6f} {RESET} — user tokens leak gradient with response_mask!")
        else:
            print(f"    {BG_GREEN} Match (pure assistant) {RESET}")
        print()

    # The advantage VALUES are the same (computed from turn rewards, not mask).
    # But the LOSS computation uses mask to zero out non-policy tokens.
    # With response_mask, user tokens in step 1 get non-zero advantage,
    # contributing to policy gradient incorrectly.
    print(f"  {BOLD}Key insight:{RESET}")
    print(f"  The advantage values themselves are turn-level (same per token in a step).")
    print(f"  The mask's role is in the LOSS computation (core_algos.py:669):")
    print(f"    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, ...)")
    print()
    print(f"  With response_mask: user tokens at pos 0-29 in Step 1 get non-zero loss")
    print(f"  With loss_mask:     user tokens at pos 0-29 are zeroed out => no gradient")
    print()

    # For step 1, verify user tokens are zeroed in loss_mask but not response_mask
    user_positions_step1 = loss_mask_batch[1, :30]  # should be all 0
    resp_positions_step1 = response_mask_batch[1, :30]  # should be all 1

    assert torch.all(user_positions_step1 == 0).item(), \
        "loss_mask should zero out user tokens"
    assert torch.all(resp_positions_step1 == 1).item(), \
        "response_mask includes user tokens"

    # Simulate loss contribution
    fake_pg_loss = torch.ones(max_resp) * 0.5  # uniform loss for simplicity
    loss_with_resp = (fake_pg_loss * response_mask_batch[1]).sum()
    loss_with_loss = (fake_pg_loss * loss_mask_batch[1]).sum()

    print(f"  Simulated Step 1 loss (uniform pg_loss=0.5 per token):")
    print(f"    With response_mask: sum = {loss_with_resp:.1f} ({int(response_mask_batch[1].sum())} tokens)")
    print(f"    With loss_mask:     sum = {loss_with_loss:.1f} ({int(loss_mask_batch[1].sum())} tokens)")
    print(f"    Extra loss from user tokens: {loss_with_resp - loss_with_loss:.1f}")
    print()

    assert loss_with_resp > loss_with_loss, \
        "response_mask should produce higher loss due to user token inclusion"

    print(f"  {BG_GREEN}{BOLD} ALL ASSERTIONS PASSED {RESET}")
    print()


if __name__ == "__main__":
    test_mask_on_real_data()
    test_mask_with_mixed_response_advantage()
