import torch
import numpy as np

# Mock internal imports for testing
import sys
import os

from verl.trainer.ppo.core_algos import compute_multiturn_grpo_advantage


def test_multiturn_grpo_advantage():
    # 2 groups (prompts), each with 2 trajectories
    # Group 0: traj 0 (length 3), traj 1 (length 2)
    # Group 1: traj 2 (length 2), traj 3 (length 2)
    turn_rewards = [
        [1.0, 0.5, 2.0],  # Group 0, traj 0
        [-1.0, 0.0],  # Group 0, traj 1
        [0.0, 1.0],  # Group 1, traj 2
        [1.0, -1.0],  # Group 1, traj 3
    ]

    turn_texts = [["A", "B", "C"], ["X", "Y"], ["M", "N"], ["P", "Q"]]

    # 4 padding tokens total sequence max length e.g. 5
    # Token mask records turn index for each valid token
    # Traj 0: turn 0 (2 tokens), turn 1 (2 tokens), turn 2 (1 token)
    # Traj 1: turn 0 (1 token), turn 1 (2 tokens)
    # Traj 2: turn 0 (2 tokens), turn 1 (2 tokens)
    # Traj 3: turn 0 (2 tokens), turn 1 (2 tokens)
    turn_token_mask = [
        [0, 0, 1, 1, 2],  # len 5
        [0, 1, 1],  # len 3
        [0, 0, 1, 1],  # len 4
        [0, 0, 1, 1],  # len 4
    ]

    max_tokens = 6
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    index = np.array([0, 0, 1, 1])
    traj_index = np.array([0, 1, 2, 3])

    adv, ret, _ = compute_multiturn_grpo_advantage(
        turn_rewards=turn_rewards,
        turn_texts=turn_texts,
        turn_token_mask=turn_token_mask,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        gamma=0.9,
        lambda_div=0.0,
        norm_adv_by_std_in_grpo=True,
    )

    print("Advantages:\n", adv)
    print("Masked regions correctly zero?", (adv[:, 5] == 0).all().item())

    # Check zero mean within group
    # However the flat normalization happens on the rewards before temporal credit assignment
    # So the mean of A should be close to 0 per group.


if __name__ == "__main__":
    test_multiturn_grpo_advantage()
