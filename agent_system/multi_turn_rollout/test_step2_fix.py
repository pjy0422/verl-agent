"""
Verification test for Step 2 (Group Normalization) dedup fix
in compute_multiturn_grpo_advantage.

Uses actual training log data from:
  examples/multiturn_grpo_trainer/run_logs/multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl

The test demonstrates:
  1. The old (buggy) code double-counts rewards when trajectories appear multiple times in the batch.
  2. The fixed code deduplicates correctly, giving each trajectory's rewards weight 1.
  3. The difference is significant when trajectories have unequal appearance counts.
"""

import json
import os
import sys
import torch
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from verl.trainer.ppo.core_algos import compute_multiturn_grpo_advantage


JSONL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../examples/multiturn_grpo_trainer/run_logs/"
    "multiturn_grpo_qwen2.5_7b_2gpu/train_rollout/1.jsonl",
)

GAMMA = 0.9
LAMBDA_DIV = 0.1
EPSILON = 1e-8


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_batch_from_jsonl(data):
    """
    Build mock batch arrays that mimic what ray_trainer passes to
    compute_multiturn_grpo_advantage.

    Each unique trajectory (by turn_rewards) gets a unique traj_uid.
    All entries share the same uid (single prompt group).
    """
    bsz = len(data)

    # Assign traj_uid: same trajectory (same turn_rewards) → same traj_uid
    reward_to_traj_uid = {}
    traj_uid_counter = 0
    traj_uids = []
    for d in data:
        key = tuple(round(r, 10) for r in d["turn_rewards"])
        if key not in reward_to_traj_uid:
            reward_to_traj_uid[key] = f"traj_{traj_uid_counter}"
            traj_uid_counter += 1
        traj_uids.append(reward_to_traj_uid[key])

    # All entries share one uid (single group)
    uids = np.array(["group_0"] * bsz, dtype=object)
    traj_uids = np.array(traj_uids, dtype=object)

    turn_rewards = [d["turn_rewards"] for d in data]
    turn_texts = [["text"] * len(d["turn_rewards"]) for d in data]

    # Build step_index: within each trajectory, assign 0, 1, 2, ...
    traj_step_counter = defaultdict(int)
    step_indices = []
    for d, tuid in zip(data, traj_uids):
        step_indices.append(traj_step_counter[tuid])
        traj_step_counter[tuid] += 1

    # Build a dummy turn_token_mask and response_mask
    max_resp_len = 50  # dummy value
    turn_token_mask = []
    for d in data:
        mask = []
        for t, _ in enumerate(d["turn_rewards"]):
            mask.extend([t] * 10)  # 10 tokens per turn
        turn_token_mask.append(mask)

    response_mask = torch.ones((bsz, max_resp_len), dtype=torch.float32)

    return {
        "turn_rewards": turn_rewards,
        "turn_texts": turn_texts,
        "turn_token_mask": turn_token_mask,
        "response_mask": response_mask,
        "index": uids,
        "traj_index": traj_uids,
        "step_index": step_indices,
    }


def compute_buggy_normalization(data):
    """
    Reproduce the OLD (buggy) behavior:
    No dedup → same trajectory's rewards added N times (N = appearance count).
    """
    bsz = len(data)
    all_rewards = []
    for d in data:
        r_hat = [r + LAMBDA_DIV * 1.0 for r in d["turn_rewards"]]
        all_rewards.extend(r_hat)

    ft = torch.tensor(all_rewards, dtype=torch.float32)
    mean_R = ft.mean().item()
    std_R = ft.std().item()
    return mean_R, std_R, len(all_rewards)


def compute_correct_normalization(data):
    """
    Correct behavior: each unique trajectory counted exactly once.
    """
    seen = set()
    all_rewards = []
    for d in data:
        key = tuple(round(r, 10) for r in d["turn_rewards"])
        if key not in seen:
            seen.add(key)
            r_hat = [r + LAMBDA_DIV * 1.0 for r in d["turn_rewards"]]
            all_rewards.extend(r_hat)

    ft = torch.tensor(all_rewards, dtype=torch.float32)
    mean_R = ft.mean().item()
    std_R = ft.std().item()
    return mean_R, std_R, len(all_rewards)


def compute_turn_advantages(turn_rewards, mean_R, std_R, gamma=GAMMA, epsilon=EPSILON):
    """Compute turn advantages for a single trajectory given group mean/std."""
    r_hat = [r + LAMBDA_DIV * 1.0 for r in turn_rewards]
    A_i = [(r - mean_R) / (std_R + epsilon) for r in r_hat]
    T = len(A_i)
    A_tilde = [0.0] * T
    A_tilde[-1] = A_i[-1]
    for t in reversed(range(T - 1)):
        A_tilde[t] = A_i[t] + gamma * A_tilde[t + 1]
    return A_tilde


def test_dedup_fix():
    """
    Verify that the fixed compute_multiturn_grpo_advantage correctly
    deduplicates trajectories in group normalization.
    """
    data = load_jsonl(JSONL_PATH)
    batch = build_batch_from_jsonl(data)

    # --- Compute with the FIXED function ---
    advantages, returns, turn_adv_fixed = compute_multiturn_grpo_advantage(
        turn_rewards=batch["turn_rewards"],
        turn_texts=batch["turn_texts"],
        turn_token_mask=batch["turn_token_mask"],
        response_mask=batch["response_mask"],
        index=batch["index"],
        traj_index=batch["traj_index"],
        step_index=batch["step_index"],
        gamma=GAMMA,
        lambda_div=LAMBDA_DIV,
        norm_adv_by_std_in_grpo=True,
    )

    # --- Compute expected (correct) normalization manually ---
    correct_mean, correct_std, correct_pool_size = compute_correct_normalization(data)
    buggy_mean, buggy_std, buggy_pool_size = compute_buggy_normalization(data)

    # Count unique trajectories
    unique_trajs = set()
    for d in data:
        unique_trajs.add(tuple(round(r, 10) for r in d["turn_rewards"]))
    n_unique = len(unique_trajs)
    n_turns = len(data[0]["turn_rewards"])

    print("=" * 70)
    print("Step 2 Group Normalization Dedup Fix Verification")
    print("=" * 70)
    print(f"\nBatch size:           {len(data)}")
    print(f"Unique trajectories:  {n_unique}")
    print(f"Turns per trajectory: {n_turns}")

    # Show appearance counts
    from collections import Counter
    traj_counts = Counter()
    for d in data:
        traj_counts[tuple(round(r, 10) for r in d["turn_rewards"])] += 1
    print(f"Appearance counts:    {sorted(traj_counts.values(), reverse=True)}")

    print(f"\n--- Buggy (no dedup) ---")
    print(f"  Pool size: {buggy_pool_size} (= {len(data)} entries x {n_turns} turns)")
    print(f"  Mean:      {buggy_mean:.8f}")
    print(f"  Std:       {buggy_std:.8f}")

    print(f"\n--- Fixed (deduped) ---")
    print(f"  Pool size: {correct_pool_size} (= {n_unique} trajectories x {n_turns} turns)")
    print(f"  Mean:      {correct_mean:.8f}")
    print(f"  Std:       {correct_std:.8f}")

    print(f"\n--- Difference ---")
    print(f"  Mean diff: {abs(buggy_mean - correct_mean):.8f}")
    print(f"  Std diff:  {abs(buggy_std - correct_std):.8f}")

    # Verify the fixed function matches manual correct computation
    first_traj_rewards = data[0]["turn_rewards"]
    expected_adv = compute_turn_advantages(first_traj_rewards, correct_mean, correct_std)
    actual_adv = turn_adv_fixed[0]

    print(f"\n--- Trajectory 0 Turn Advantages ---")
    print(f"  Expected (correct): {[round(a, 6) for a in expected_adv]}")
    print(f"  Actual (fixed fn):  {[round(a, 6) for a in actual_adv]}")

    buggy_adv = compute_turn_advantages(first_traj_rewards, buggy_mean, buggy_std)
    print(f"  Buggy (old code):   {[round(a, 6) for a in buggy_adv]}")

    # Assert fixed function matches expected
    for t in range(n_turns):
        assert abs(actual_adv[t] - expected_adv[t]) < 1e-5, (
            f"Turn {t}: actual={actual_adv[t]:.8f} != expected={expected_adv[t]:.8f}"
        )
    print("\n  [PASS] Fixed function matches expected computation.")

    # Show that different trajectories also match
    print(f"\n--- All Unique Trajectories ---")
    verified_trajs = set()
    all_pass = True
    for i, d in enumerate(data):
        key = tuple(round(r, 10) for r in d["turn_rewards"])
        if key in verified_trajs:
            continue
        verified_trajs.add(key)

        expected = compute_turn_advantages(d["turn_rewards"], correct_mean, correct_std)
        actual = turn_adv_fixed[i]

        match = all(abs(actual[t] - expected[t]) < 1e-5 for t in range(len(expected)))
        status = "PASS" if match else "FAIL"
        if not match:
            all_pass = False
        print(
            f"  Traj {len(verified_trajs):2d} (idx={i:2d}): "
            f"rewards={[round(r, 3) for r in d['turn_rewards']]} "
            f"→ adv={[round(a, 4) for a in actual]} [{status}]"
        )

    assert all_pass, "Some trajectories did not match!"
    print(f"\n  [PASS] All {n_unique} unique trajectories verified.")

    # Demonstrate the impact: how much do advantages differ between buggy and fixed?
    print(f"\n--- Impact Analysis ---")
    max_diff = 0
    for i, d in enumerate(data):
        buggy = compute_turn_advantages(d["turn_rewards"], buggy_mean, buggy_std)
        fixed = turn_adv_fixed[i]
        for t in range(len(fixed)):
            diff = abs(buggy[t] - fixed[t])
            if diff > max_diff:
                max_diff = diff
    print(f"  Max advantage difference (buggy vs fixed): {max_diff:.6f}")
    print(f"  This difference directly affects PPO policy gradient updates.")

    # Verify that entries with same traj_uid get same turn_advantages
    print(f"\n--- Consistency Check ---")
    traj_to_advs = defaultdict(list)
    for i, d in enumerate(data):
        key = tuple(round(r, 10) for r in d["turn_rewards"])
        traj_to_advs[key].append(turn_adv_fixed[i])

    consistent = True
    for key, advs_list in traj_to_advs.items():
        for j in range(1, len(advs_list)):
            for t in range(len(advs_list[0])):
                if abs(advs_list[0][t] - advs_list[j][t]) > 1e-6:
                    consistent = False
                    break
    assert consistent, "Same trajectory should get same turn_advantages!"
    print(f"  [PASS] All duplicate entries of same trajectory get identical advantages.")

    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_dedup_fix()
