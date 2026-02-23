import pytest
import torch
import numpy as np
from verl import DataProto
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from transformers import AutoTokenizer


class MockConfig:
    class Data:
        max_prompt_length = 512
        truncation = "right"
        train_batch_size = 2

        def get(self, key, default=None):
            return default

    class Env:
        max_steps = 3

        class Rollout:
            n = 1

        rollout = Rollout()

    class Algorithm:
        class FilterGroups:
            enable = False
            max_num_gen_batches = 1

    data = Data()
    env = Env()
    algorithm = Algorithm()


class MockEnvManager(EnvironmentManagerBase):
    def __init__(self, num_envs=2, max_steps=3):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.current_steps = np.zeros(num_envs, dtype=int)

    def reset(self, kwargs=None):
        self.current_steps = np.zeros(self.num_envs, dtype=int)
        obs = {
            "text": ["Initial Observation 1", "Initial Observation 2"],
            "image": None,
            "anchor": None,
        }
        infos = [{"is_action_valid": True} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, actions):
        self.current_steps += 1

        # dummy observations back
        next_obs = {
            "text": [
                f"Obs turn {self.current_steps[i]}"
                for i in range(self.num_envs)
            ],
            "image": None,
            "anchor": None,
        }

        # simulate step reward: turn 1 = 0.5, turn 2 = 1.0, turn 3 = 1.5
        rewards = np.array(
            [0.5 * step for step in self.current_steps], dtype=np.float32
        )

        # done condition
        dones = self.current_steps >= self.max_steps

        infos = [{"is_action_valid": True} for _ in range(self.num_envs)]

        return next_obs, rewards, dones, infos

    def success_evaluator(
        self, total_infos, total_batch_list, episode_rewards, episode_lengths
    ):
        return {"success_rate": np.ones(self.num_envs, dtype=bool)}


class MockActorRolloutWG:
    def __init__(self, world_size=1):
        self.world_size = world_size

        # This is a dummy response tensor.
        # Format (batch_size, seq_length).
        # We will hardcode 5 tokens for realism.
        self.seq_len = 5

    def generate_sequences(self, batch: DataProto):
        # Mocks generating sequences and appending to batch
        batch_size = len(batch.batch["input_ids"])

        responses = torch.ones((batch_size, self.seq_len), dtype=torch.long)
        batch.batch["responses"] = responses

        return batch


def test_vanilla_multi_turn_loop():
    config = MockConfig()
    # Simple tokenizer (use padding logic as required)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collector = TrajectoryCollector(
        config=config, tokenizer=tokenizer, processor=None
    )

    envs = MockEnvManager(num_envs=2, max_steps=3)
    actor_rollout_wg = MockActorRolloutWG()

    # Initialize gen_batch
    raw_prompts = ["Hello env 1", "Hello env 2"]
    data_sources = ["source_a", "source_b"]

    # Since batch generation uses tensor operations
    batch_dict = {"input_ids": torch.randint(0, 1000, (2, 10))}
    non_tensor_batch_dict = {
        "raw_prompt": np.array(raw_prompts, dtype=object),
        "data_source": np.array(data_sources, dtype=object),
    }
    gen_batch = DataProto.from_dict(
        tensors=batch_dict, non_tensors=non_tensor_batch_dict
    )

    # Run the loop
    (
        total_batch_list,
        episode_rewards,
        episode_lengths,
        success,
        traj_uid,
        tool_callings,
        turn_rewards_all,
        turn_texts_all,
        turn_token_mask_all,
    ) = collector.vanilla_multi_turn_loop(
        gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs
    )

    # Extract Dataproto
    out_dp = collector.gather_rollout_data(
        total_batch_list=total_batch_list,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success=success,
        traj_uid=traj_uid,
        tool_callings=tool_callings,
        turn_rewards_all=turn_rewards_all,
        turn_texts_all=turn_texts_all,
        turn_token_mask_all=turn_token_mask_all,
    )

    # Verifications
    num_envs = 2
    max_steps = 3

    # DataProto checks
    assert (
        len(out_dp.batch["input_ids"]) == num_envs * max_steps
    ), "Each turn should generate one batch entry"

    # We can inspect the non_tensor_batch to check if mapping rules hold
    turn_rewards_from_dp = out_dp.non_tensor_batch["turn_rewards"]
    turn_texts_from_dp = out_dp.non_tensor_batch["turn_texts"]
    turn_token_masks_from_dp = out_dp.non_tensor_batch["turn_token_mask"]
    episode_lengths_from_dp = out_dp.non_tensor_batch["episode_lengths"]

    # Loop over all individual trajectory components to verify correctness
    for i in range(len(turn_rewards_from_dp)):
        env_turn_rewards = turn_rewards_from_dp[i]  # array of length 3
        env_turn_texts = turn_texts_from_dp[i]  # array of length 3
        env_episode_length = episode_lengths_from_dp[i]  # scalar = 3
        env_turn_token_mask = turn_token_masks_from_dp[i]

        # Verify sizes match episode_length
        assert (
            len(env_turn_rewards) == env_episode_length
        ), f"Reward len {len(env_turn_rewards)} != eps len {env_episode_length}"
        assert (
            len(env_turn_texts) == env_episode_length
        ), f"Text len {len(env_turn_texts)} != eps len {env_episode_length}"

        # Verify Token mask sums to (episode_length * generated_tokens)
        expected_total_tokens = env_episode_length * actor_rollout_wg.seq_len
        assert (
            len(env_turn_token_mask) == expected_total_tokens
        ), f"Mask len {len(env_turn_token_mask)} != expected {expected_total_tokens}"

        # Verify specific mock reward values (Turn 1: 0.5, Turn 2: 1.0, Turn 3: 1.5)
        # Note: In DP, the data is replicated for all turns of the same trajectory, so we just check content
        assert np.isclose(env_turn_rewards[0], 0.5)
        assert np.isclose(env_turn_rewards[1], 1.0)
        assert np.isclose(env_turn_rewards[2], 1.5)

    print("\n" + "=" * 50)
    print("✨ Unit Test Passed Perfectly ✨")
    print("=" * 50)

    print("\nDetailed Summary for Environment 0 (First Batch Element):")
    print("-" * 50)
    print(f"Episode Length (Turns)  : {episode_lengths_from_dp[0]}")
    print(f"Collected Turn Rewards  : {turn_rewards_from_dp[0]}")
    print(f"Collected Turn Texts    : {turn_texts_from_dp[0]}")
    print(f"Turn Token Mask Layout  : {turn_token_masks_from_dp[0]}")

    print(
        "\nToken Mask verifies that each generated token directly maps back to the turn id (_step)"
    )
    print("-" * 50)


if __name__ == "__main__":
    test_vanilla_multi_turn_loop()
