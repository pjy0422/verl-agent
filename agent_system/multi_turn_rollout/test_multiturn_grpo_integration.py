import torch
import numpy as np

# Mock internal imports for testing
import sys
import os

from verl.trainer.ppo.core_algos import compute_multiturn_grpo_advantage
from verl.trainer.ppo.ray_trainer import compute_advantage, AdvantageEstimator
from verl import DataProto


def test_multiturn_grpo_advantage_ray_trainer_integration():
    # Construct a mock DataProto that mimics what is generated in rollout
    batch_size = 4

    # 2 groups (prompts), each with 2 trajectories
    turn_rewards = np.array(
        [
            [1.0, 0.5, 2.0],  # Group 0, traj 0
            [-1.0, 0.0],  # Group 0, traj 1
            [0.0, 1.0],  # Group 1, traj 2
            [1.0, -1.0],  # Group 1, traj 3
        ],
        dtype=object,
    )

    turn_texts = np.array(
        [["A", "B", "C"], ["X", "Y"], ["M", "N"], ["P", "Q"]], dtype=object
    )

    turn_token_mask = np.array(
        [
            [0, 0, 1, 1, 2],  # len 5
            [0, 1, 1],  # len 3
            [0, 0, 1, 1],  # len 4
            [0, 0, 1, 1],  # len 4
        ],
        dtype=object,
    )

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

    # Normally, token_level_rewards holds single-scalar values for GRPO,
    # but for multi-turn it's mostly ignored in favor of `turn_rewards`
    token_level_rewards = torch.zeros((batch_size, max_tokens))

    index = np.array([0, 0, 1, 1])
    traj_index = np.array([0, 1, 2, 3])

    # Assemble DataProto
    data = DataProto.from_single_dict(
        {
            "response_mask": response_mask,
            "token_level_rewards": token_level_rewards,  # not strictly used in MULTITURN_GRPO but queried
        }
    )
    data.non_tensor_batch["turn_rewards"] = turn_rewards
    data.non_tensor_batch["turn_texts"] = turn_texts
    data.non_tensor_batch["turn_token_mask"] = turn_token_mask
    data.non_tensor_batch["uid"] = index
    data.non_tensor_batch["traj_uid"] = traj_index

    # Pass custom lambda_div
    data_out = compute_advantage(
        data=data,
        adv_estimator=AdvantageEstimator.MULTITURN_GRPO,
        gamma=0.9,  # gamma passed from config
        lambda_div=0.5,  # Custom lambda_div test parameter
        norm_adv_by_std_in_grpo=True,
    )

    adv = data_out.batch["advantages"]
    returns = data_out.batch["returns"]

    print("Integration test DataProto advantages:\n", adv)
    print(
        "Does returned tensor match masked regions exactly?: ",
        (adv[:, 5] == 0).all().item(),
    )
    print("Do returns equal advantages?: ", torch.equal(adv, returns))


if __name__ == "__main__":
    test_multiturn_grpo_advantage_ray_trainer_integration()
