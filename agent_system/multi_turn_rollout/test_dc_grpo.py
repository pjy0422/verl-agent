"""
DC-GRPO (Decomposed Credit GRPO) formula verification tests.

Verifies that compute_dc_grpo_advantage matches the mathematical formulas
from dc_grpo_v2.md step by step.

Test cases:
  A: Refusal trajectory (from dc_grpo_v2.md "Why DC-GRPO Solves the Refusal Problem")
  B: Single-turn equivalence (dc_grpo_v2.md "Single-Turn Equivalence")
  C: Exact decomposition identity (dc_grpo_v2.md "Exact Decomposition")
  D: Last-turn formula
  E: Variable-length trajectories
"""

import sys
import math

import numpy as np
import torch

sys.path.insert(0, "/home2/pjy0422/workspace/verl-agent")
from verl.trainer.ppo.core_algos import (
    compute_dc_grpo_advantage,
    compute_grpo_outcome_advantage,
)


# ─── Helpers ──────────────────────────────────────────────────────────
def _make_inputs(
    reward_lists: list[list[float]],
    uid_list: list,
    traj_uid_list: list,
    step_indices: list[int],
    max_tokens: int = 8,
):
    """Build inputs for compute_dc_grpo_advantage from compact reward lists."""
    bsz = len(reward_lists)
    response_mask = torch.ones((bsz, max_tokens), dtype=torch.float32)
    turn_texts = [["dummy"] * len(r) for r in reward_lists]
    turn_token_mask = [list(range(len(r))) for r in reward_lists]
    return dict(
        turn_rewards=reward_lists,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask,
        index=np.array(uid_list),
        traj_index=np.array(traj_uid_list),
        step_index=step_indices,
    )


def _compute_returns(rewards, gamma):
    """R_{i,t} = r_{i,t} + γ · R_{i,t+1}, R_{i,T+1}=0"""
    T = len(rewards)
    R = [0.0] * (T + 1)
    for t in reversed(range(T)):
        R[t] = rewards[t] + gamma * R[t + 1]
    return R[:T]


# ─── Test A: Refusal trajectory ──────────────────────────────────────
def test_refusal_trajectory():
    """
    From dc_grpo_v2.md "Why DC-GRPO Solves the Refusal Problem":
      Trajectory 1 (refusal):   rewards = [-1, 1, 1, 1, 1]
      Trajectory 2 (compliant): rewards = [ 1, 1, 1, 1, 1]
      gamma = 0.99

    Verify:
      1. R_{1,1} for refusal traj
      2. μ^r_1 = mean(-1, 1) = 0.0
      3. σ^r_1 = std(-1, 1)
      4. I_{1,1} = (-1 - 0) / (σ^r_1 + ε) → LARGE NEGATIVE
    """
    gamma = 0.99
    eps = 1e-8
    rewards_1 = [-1.0, 1.0, 1.0, 1.0, 1.0]  # refusal
    rewards_2 = [1.0, 1.0, 1.0, 1.0, 1.0]  # compliant
    T = 5

    # Manual computation
    R_1 = _compute_returns(rewards_1, gamma)
    R_2 = _compute_returns(rewards_2, gamma)

    print("=== Test A: Refusal trajectory ===")
    print(f"  R_1 (refusal)  = {[f'{x:.4f}' for x in R_1]}")
    print(f"  R_2 (compliant)= {[f'{x:.4f}' for x in R_2]}")

    # Per-turn stats
    A_manual = [0.0] * T
    for t in range(T):
        r_at_t = [rewards_1[t], rewards_2[t]]
        mu_r = sum(r_at_t) / 2
        sigma_r = torch.tensor(r_at_t).std().item()

        I_1t = (rewards_1[t] - mu_r) / (sigma_r + eps)

        if t < T - 1:
            R_future = [R_1[t + 1], R_2[t + 1]]
            mu_R = sum(R_future) / 2
            sigma_R = torch.tensor(R_future).std().item()
            F_1t = gamma * (R_1[t + 1] - mu_R) / (sigma_R + eps)
            A_manual[t] = I_1t + F_1t
        else:
            A_manual[t] = I_1t

        label = f"  Turn {t}: μ^r={mu_r:.4f}, σ^r={sigma_r:.4f}, I_{t}={I_1t:.4f}"
        if t < T - 1:
            label += f", F_{t}={F_1t:.4f}"
        print(label)

    print(f"  A_manual (refusal, traj 1) = {[f'{x:.4f}' for x in A_manual]}")

    # Call implementation — each batch entry sees ONE step
    # Build batch: 10 entries (5 steps x 2 trajs), each with full trajectory rewards
    reward_lists = []
    uid_list = []
    traj_uid_list = []
    step_indices = []
    for step in range(T):
        reward_lists.append(rewards_1)
        uid_list.append("prompt_0")
        traj_uid_list.append("traj_0")
        step_indices.append(step)
    for step in range(T):
        reward_lists.append(rewards_2)
        uid_list.append("prompt_0")
        traj_uid_list.append("traj_1")
        step_indices.append(step)

    inputs = _make_inputs(reward_lists, uid_list, traj_uid_list, step_indices)
    token_adv, _, A_dc = compute_dc_grpo_advantage(
        **inputs, gamma=gamma, lambda_div=0.0, epsilon=eps
    )

    # A_dc[0..4] corresponds to traj_0 (refusal), steps 0..4
    for t in range(T):
        actual = A_dc[t][t]  # step t, turn t
        expected = A_manual[t]
        print(f"  Step {t}: expected={expected:.6f}, actual={actual:.6f}")
        assert abs(actual - expected) < 1e-5, (
            f"Mismatch at turn {t}: expected={expected}, actual={actual}"
        )

    # Key property: I_{1,1} is large negative (immediate penalty survives)
    mu_r_0 = 0.0  # mean(-1, 1)
    sigma_r_0 = torch.tensor([-1.0, 1.0]).std().item()
    I_11 = (-1.0 - mu_r_0) / (sigma_r_0 + eps)
    assert I_11 < -0.5, f"Expected large negative I_11, got {I_11}"
    print(f"  I_{{1,1}} = {I_11:.4f} (large negative → immediate penalty preserved)")
    print("  PASSED\n")


# ─── Test B: Single-turn equivalence ─────────────────────────────────
def test_single_turn_equivalence():
    """
    dc_grpo_v2.md "Single-Turn Equivalence":
      T=1: A_{i,1} = (r_{i,1} - μ^r_1) / (σ^r_1 + ε) + 0 = standard GRPO advantage
    """
    print("=== Test B: Single-turn equivalence ===")
    eps = 1e-6
    rewards = [0.5, -0.3, 1.2, 0.0, 0.8]
    G = len(rewards)

    # Manual single-turn DC-GRPO
    mu = sum(rewards) / G
    sigma = torch.tensor(rewards).std().item()
    expected = [(r - mu) / (sigma + eps) for r in rewards]

    # Call DC-GRPO with T=1 trajectories
    reward_lists = [[r] for r in rewards]
    uid_list = ["prompt_0"] * G
    traj_uid_list = [f"traj_{i}" for i in range(G)]
    step_indices = [0] * G
    inputs = _make_inputs(reward_lists, uid_list, traj_uid_list, step_indices)
    token_adv, _, A_dc = compute_dc_grpo_advantage(
        **inputs, gamma=0.99, lambda_div=0.0, epsilon=eps
    )

    for i in range(G):
        actual = A_dc[i][0]
        print(f"  Traj {i}: r={rewards[i]:.2f}, expected={expected[i]:.6f}, actual={actual:.6f}")
        assert abs(actual - expected[i]) < 1e-5, (
            f"Mismatch at traj {i}: expected={expected[i]}, actual={actual}"
        )

    # Also compare with GRPO outcome advantage
    token_rewards = torch.zeros((G, 8))
    for i in range(G):
        token_rewards[i, -1] = rewards[i]
    resp_mask = torch.ones((G, 8))
    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_rewards,
        response_mask=resp_mask,
        index=np.array(uid_list),
        traj_index=np.array(traj_uid_list),
        epsilon=eps,
        norm_adv_by_std_in_grpo=True,
    )
    # GRPO advantage is uniform across tokens; check the value
    for i in range(G):
        grpo_val = grpo_adv[i, 0].item()
        dc_val = A_dc[i][0]
        print(f"  Traj {i}: DC-GRPO={dc_val:.6f}, GRPO={grpo_val:.6f}")
        assert abs(dc_val - grpo_val) < 1e-4, (
            f"DC-GRPO != GRPO at traj {i}: dc={dc_val}, grpo={grpo_val}"
        )

    print("  PASSED\n")


# ─── Test C: Exact decomposition identity ────────────────────────────
def test_exact_decomposition():
    """
    dc_grpo_v2.md "Exact Decomposition":
      R_{i,t} - μ^R_t = (r_{i,t} - μ^r_t) + γ · (R_{i,t+1} - μ^R_{t+1})
    """
    print("=== Test C: Exact decomposition identity ===")
    gamma = 0.95
    rewards_list = [
        [2.0, -1.0, 3.0, 0.5],
        [-0.5, 2.0, 1.0, -1.0],
        [1.0, 0.0, 0.0, 2.0],
    ]
    G = len(rewards_list)
    T = 4

    # Compute returns
    returns = [_compute_returns(r, gamma) for r in rewards_list]

    # Compute per-turn stats
    mu_r = []
    mu_R = []
    for t in range(T):
        mu_r.append(sum(rewards_list[j][t] for j in range(G)) / G)
        mu_R.append(sum(returns[j][t] for j in range(G)) / G)

    # Boundary: μ^R_{T+1} = 0 (all returns past T are 0)
    mu_R.append(0.0)

    # Verify identity for each trajectory and turn
    for i in range(G):
        R_i = returns[i] + [0.0]  # append R_{i,T+1}=0
        for t in range(T):
            lhs = R_i[t] - mu_R[t]
            rhs = (rewards_list[i][t] - mu_r[t]) + gamma * (R_i[t + 1] - mu_R[t + 1])
            print(f"  Traj {i}, Turn {t}: LHS={lhs:.6f}, RHS={rhs:.6f}")
            assert abs(lhs - rhs) < 1e-10, (
                f"Decomposition identity failed at i={i}, t={t}: LHS={lhs}, RHS={rhs}"
            )

    print("  PASSED\n")


# ─── Test D: Last-turn formula ────────────────────────────────────────
def test_last_turn():
    """
    For last turn t=T: A_{i,T} = (r_{i,T} - μ^r_T) / (σ^r_T + ε)
    No future term. F_{i,T} = 0.
    """
    print("=== Test D: Last-turn formula ===")
    gamma = 0.99
    eps = 1e-8
    rewards_1 = [1.0, -2.0, 3.0]
    rewards_2 = [0.0, 1.0, -1.0]
    T = 3

    # Build batch: 2 trajs, but only check the LAST turn
    reward_lists = []
    uid_list = []
    traj_uid_list = []
    step_indices = []
    for step in range(T):
        reward_lists.append(rewards_1)
        uid_list.append("p0")
        traj_uid_list.append("t0")
        step_indices.append(step)
    for step in range(T):
        reward_lists.append(rewards_2)
        uid_list.append("p0")
        traj_uid_list.append("t1")
        step_indices.append(step)

    inputs = _make_inputs(reward_lists, uid_list, traj_uid_list, step_indices)
    _, _, A_dc = compute_dc_grpo_advantage(
        **inputs, gamma=gamma, lambda_div=0.0, epsilon=eps
    )

    # Manual: last turn (t=2)
    r_last = [rewards_1[2], rewards_2[2]]  # [3.0, -1.0]
    mu_r_last = sum(r_last) / 2
    sigma_r_last = torch.tensor(r_last).std().item()

    for i, (rewards, traj_name) in enumerate([(rewards_1, "traj_0"), (rewards_2, "traj_1")]):
        base_idx = i * T
        # The last-turn entry
        last_step_idx = T - 1
        actual = A_dc[base_idx + last_step_idx][last_step_idx]
        expected = (rewards[last_step_idx] - mu_r_last) / (sigma_r_last + eps)
        print(f"  {traj_name} last turn: r={rewards[last_step_idx]}, expected={expected:.6f}, actual={actual:.6f}")
        assert abs(actual - expected) < 1e-5, (
            f"Last-turn mismatch for {traj_name}: expected={expected}, actual={actual}"
        )

    # Verify F_{i,T} = 0 implicitly: advantage at last turn is purely I_{i,T}
    # (already verified above since expected only uses immediate term)
    print("  F_{i,T} = 0 confirmed (advantage is purely immediate at last turn)")
    print("  PASSED\n")


# ─── Test E: Variable-length trajectories ─────────────────────────────
def test_variable_length():
    """
    Group with trajectories of different lengths (T=3 and T=5).
    Per-turn stats only include trajectories that reached that turn.
    """
    print("=== Test E: Variable-length trajectories ===")
    gamma = 0.9
    eps = 1e-8
    rewards_short = [1.0, -1.0, 2.0]  # T=3
    rewards_long = [0.5, 0.5, 0.5, 1.0, -0.5]  # T=5

    # Build batch: 2 trajs of different lengths
    reward_lists = []
    uid_list = []
    traj_uid_list = []
    step_indices = []

    for step in range(len(rewards_short)):
        reward_lists.append(rewards_short)
        uid_list.append("p0")
        traj_uid_list.append("t0")
        step_indices.append(step)
    for step in range(len(rewards_long)):
        reward_lists.append(rewards_long)
        uid_list.append("p0")
        traj_uid_list.append("t1")
        step_indices.append(step)

    inputs = _make_inputs(reward_lists, uid_list, traj_uid_list, step_indices, max_tokens=8)
    _, _, A_dc = compute_dc_grpo_advantage(
        **inputs, gamma=gamma, lambda_div=0.0, epsilon=eps
    )

    # Manual computation
    R_short = _compute_returns(rewards_short, gamma)
    R_long = _compute_returns(rewards_long, gamma)

    T_short = len(rewards_short)
    T_long = len(rewards_long)
    max_T = max(T_short, T_long)

    # Per-turn stats: turns 0,1,2 have both trajs; turns 3,4 only long traj
    mu_r = []
    sigma_r = []
    mu_R = []
    sigma_R = []
    for t in range(max_T):
        r_at_t = []
        R_at_t = []
        if t < T_short:
            r_at_t.append(rewards_short[t])
            R_at_t.append(R_short[t])
        if t < T_long:
            r_at_t.append(rewards_long[t])
            R_at_t.append(R_long[t])

        mu_r.append(sum(r_at_t) / len(r_at_t))
        sigma_r.append(torch.tensor(r_at_t).std().item() if len(r_at_t) > 1 else 0.0)
        mu_R.append(sum(R_at_t) / len(R_at_t))
        sigma_R.append(torch.tensor(R_at_t).std().item() if len(R_at_t) > 1 else 0.0)

    print(f"  mu_r   = {[f'{x:.4f}' for x in mu_r]}")
    print(f"  sigma_r= {[f'{x:.4f}' for x in sigma_r]}")

    # Verify short trajectory (traj_0)
    R_short_full = R_short + [0.0]  # boundary
    expected_short = []
    for t in range(T_short):
        I_t = (rewards_short[t] - mu_r[t]) / (sigma_r[t] + eps)
        if t < T_short - 1:
            F_t = gamma * (R_short_full[t + 1] - mu_R[t + 1]) / (sigma_R[t + 1] + eps)
            expected_short.append(I_t + F_t)
        else:
            expected_short.append(I_t)

    for t in range(T_short):
        actual = A_dc[t][t]  # batch entry t, step t
        print(f"  Short traj, Turn {t}: expected={expected_short[t]:.6f}, actual={actual:.6f}")
        assert abs(actual - expected_short[t]) < 1e-5, (
            f"Mismatch at short traj turn {t}"
        )

    # Verify long trajectory (traj_1, starts at index T_short in batch)
    R_long_full = R_long + [0.0]
    expected_long = []
    for t in range(T_long):
        I_t = (rewards_long[t] - mu_r[t]) / (sigma_r[t] + eps)
        if t < T_long - 1:
            F_t = gamma * (R_long_full[t + 1] - mu_R[t + 1]) / (sigma_R[t + 1] + eps)
            expected_long.append(I_t + F_t)
        else:
            expected_long.append(I_t)

    for t in range(T_long):
        actual = A_dc[T_short + t][t]  # batch entry T_short+t, step t
        print(f"  Long traj, Turn {t}: expected={expected_long[t]:.6f}, actual={actual:.6f}")
        assert abs(actual - expected_long[t]) < 1e-5, (
            f"Mismatch at long traj turn {t}"
        )

    # Turns 3,4: only 1 trajectory → σ^r=0, σ^R=0 → epsilon handles it
    for t in range(T_short, T_long):
        assert sigma_r[t] == 0.0, f"Expected sigma_r[{t}]=0 for single-traj turn"
    print("  Turns 3-4 have σ^r=0 (single trajectory) → epsilon handles correctly")
    print("  PASSED\n")


# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_refusal_trajectory()
    test_single_turn_equivalence()
    test_exact_decomposition()
    test_last_turn()
    test_variable_length()
    print("=" * 50)
    print("ALL DC-GRPO TESTS PASSED")
    print("=" * 50)
