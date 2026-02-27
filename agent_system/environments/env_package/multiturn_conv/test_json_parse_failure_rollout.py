"""
Test: JSON 파싱 실패 시 멀티턴 환경 롤아웃 검증

JSON parsing 에 실패해서 fallback prompt 가 사용되고, invalid action penalty 가
적용되는 전체 흐름을 mocking 기반으로 테스트한다.

검증 항목:
  1. projection: 다양한 malformed 입력 → valid=0, fallback prompt 추출
  2. env_manager: step() 에서 is_action_valid=0 이 infos 에 정상 기록
  3. env_manager: 파싱 실패해도 멀티턴 history 누적이 정상 동작
  4. penalty: apply_invalid_action_penalty 가 reward 를 올바르게 감소시킴
"""

import sys
import os
import json
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, AsyncMock
from functools import partial
from collections import defaultdict

# Ensure project root is on path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
)

from agent_system.environments.env_package.multiturn_conv.projection import (
    parse_json_response,
    _extract_fallback_prompt,
    multiturn_conv_projection,
)
from agent_system.environments.env_manager import (
    MultiTurnConvEnvironmentManager,
)
from agent_system.environments.base import to_numpy


# =====================================================================
# 1. projection 단위 테스트 — 파싱 실패 케이스
# =====================================================================
class TestParseJsonResponseFailure:
    """parse_json_response 가 None 을 반환하는 malformed 입력들."""

    def test_empty_string(self):
        assert parse_json_response("") is None

    def test_none_input(self):
        assert parse_json_response(None) is None

    def test_plain_text_no_json(self):
        assert parse_json_response("This is just a plain text response") is None

    def test_broken_json_no_closing_brace(self):
        assert parse_json_response('{"prompt": "hello') is None

    def test_only_think_tags(self):
        assert parse_json_response("<think>some reasoning</think>") is None

    def test_think_then_garbage(self):
        result = parse_json_response("<think>reasoning</think>not json at all")
        assert result is None

    def test_markdown_block_with_invalid_json(self):
        text = '```json\n{invalid json content}\n```'
        assert parse_json_response(text) is None


class TestParseJsonResponsePartial:
    """JSON 은 파싱되지만 "prompt" 키가 없는 경우."""

    def test_json_without_prompt_key(self):
        text = '{"rationale": "some reasoning", "message": "hello"}'
        result = parse_json_response(text)
        assert result is not None
        assert "prompt" not in result

    def test_json_with_only_rationale(self):
        text = '{"rationale": "I will try a different approach"}'
        result = parse_json_response(text)
        assert result is not None
        assert "prompt" not in result


class TestFallbackPromptExtraction:
    """_extract_fallback_prompt 의 다양한 fallback 경로."""

    def test_malformed_json_with_prompt_key_regex(self):
        """Fallback 1: regex 로 "prompt" 값을 추출."""
        action = '{"rationale": "abc", "prompt": some unquoted value here}'
        result = _extract_fallback_prompt(action, parsed=None)
        assert "some unquoted value here" in result

    def test_truncated_json_with_prompt(self):
        """Fallback 1: 잘린 JSON 에서 prompt 값 추출."""
        action = '{"rationale": "abc", "prompt": "this is my attack prompt that got cut off...'
        result = _extract_fallback_prompt(action, parsed=None)
        assert "this is my attack prompt" in result

    def test_markdown_bold_prompt(self):
        """Fallback 1: markdown **prompt**: 형태."""
        action = "**prompt**: Here is my carefully crafted attack"
        result = _extract_fallback_prompt(action, parsed=None)
        assert "carefully crafted attack" in result

    def test_json_with_alternative_key(self):
        """Fallback 2: prompt 키 없지만 alternative key 있는 경우."""
        parsed = {"rationale": "test", "attack_prompt": "my attack content"}
        result = _extract_fallback_prompt("some text", parsed=parsed)
        assert result == "my attack content"

    def test_json_with_longest_string(self):
        """Fallback 2: 모든 alternative key 없으면 가장 긴 문자열."""
        parsed = {
            "rationale": "short",
            "context": "This is a very long string that should be selected as fallback because it is the longest",
        }
        result = _extract_fallback_prompt("some text", parsed=parsed)
        assert "very long string" in result

    def test_raw_text_fallback(self):
        """Fallback 3: JSON 파싱 완전 실패, raw text 사용."""
        action = "Just a plain text attack prompt that is long enough"
        result = _extract_fallback_prompt(action, parsed=None)
        assert len(result) > 10

    def test_absolute_fallback(self):
        """Fallback 4: 아무것도 없으면 generic continuation."""
        result = _extract_fallback_prompt("", parsed=None)
        assert result == "Continue the conversation."

    def test_think_tag_stripped_before_fallback(self):
        """<think> 태그가 제거된 후 fallback 진행."""
        action = '<think>reasoning here</think>{"rationale": "abc", "prompt": unquoted_prompt_text}'
        result = _extract_fallback_prompt(action, parsed=None)
        assert "unquoted_prompt_text" in result


class TestMultiturnConvProjection:
    """multiturn_conv_projection 통합 테스트 — 파싱 실패 시 valid=0."""

    def test_all_invalid_actions(self):
        """모든 액션이 파싱 실패 → 전부 valid=0."""
        actions = [
            "This is plain text, not JSON",
            "<think>reasoning</think>also not json",
            "",
        ]
        results, valids = multiturn_conv_projection(actions)
        assert len(results) == 3
        assert len(valids) == 3
        assert all(v == 0 for v in valids)
        # results 는 항상 비어있지 않아야 함
        assert all(len(r) > 0 for r in results)

    def test_mixed_valid_and_invalid(self):
        """valid + invalid 혼합."""
        actions = [
            '{"rationale": "good", "prompt": "attack prompt here"}',  # valid
            "just plain text with no json at all",  # invalid
            '{"rationale": "partial"}',  # json OK but no "prompt" key
        ]
        results, valids = multiturn_conv_projection(actions)
        assert valids == [1, 0, 0]
        assert results[0] == "attack prompt here"
        assert len(results[1]) > 0  # fallback 사용
        assert len(results[2]) > 0  # fallback 사용

    def test_truncated_json_is_invalid(self):
        """잘린 JSON → valid=0 + fallback."""
        actions = ['{"rationale": "abc", "prompt": "my prompt that got cut off...']
        results, valids = multiturn_conv_projection(actions)
        assert valids == [0]
        assert "my prompt" in results[0]  # fallback 으로 추출

    def test_markdown_wrapped_invalid_json(self):
        """markdown code block 안에 invalid JSON."""
        actions = ['```json\n{broken: json, no quotes}\n```']
        results, valids = multiturn_conv_projection(actions)
        assert valids == [0]
        assert len(results[0]) > 0

    def test_answer_tag_with_no_json(self):
        """<answer> 태그 안에 JSON 없는 경우."""
        actions = ["<answer>Just some reasoning without json</answer>"]
        results, valids = multiturn_conv_projection(actions)
        assert valids == [0]


# =====================================================================
# 2. MultiTurnConvEnvironmentManager — 파싱 실패 시 rollout 통합 테스트
# =====================================================================
class MockMultiTurnConvEnv:
    """MultiTurnConvEnv 를 mocking — 외부 vLLM 서버 없이 테스트."""

    def __init__(self, batch_size: int, max_steps: int = 3):
        self.batch_size = batch_size
        self.max_steps = max_steps
        self._turn = 0

    def reset(self, kwargs):
        self._turn = 0
        obs = ["" for _ in range(len(kwargs))]
        infos = [
            {"behavior": kw.get("behavior", ""), "data_source": "test"}
            for kw in kwargs
        ]
        return obs, infos

    def step(self, actions):
        self._turn += 1
        obs = [f"Target response to: {a[:50]}" for a in actions]
        rewards = [0.3 for _ in actions]
        dones = [self._turn >= self.max_steps for _ in actions]
        infos = [
            {
                "turn": self._turn,
                "target_response": obs[i],
                "safety_unsafe_prob": 0.3,
                "refusal_complied": True,
                "won": False,
                "data_source": "test",
            }
            for i in range(len(actions))
        ]
        return obs, rewards, dones, infos


def _make_config(max_steps=3):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {"train_batch_size": 2},
            "env": {"env_name": "multiturn_conv", "max_steps": max_steps},
        }
    )


class TestEnvManagerParseFailureRollout:
    """파싱 실패 액션으로 멀티턴 rollout 이 정상 동작하는지 검증."""

    def setup_method(self):
        self.config = _make_config(max_steps=3)
        self.mock_env = MockMultiTurnConvEnv(batch_size=2, max_steps=3)
        self.env_manager = MultiTurnConvEnvironmentManager(
            envs=self.mock_env,
            projection_f=multiturn_conv_projection,
            config=self.config,
        )

    def test_parse_failure_sets_is_action_valid_to_zero(self):
        """파싱 실패 시 info['is_action_valid'] == 0."""
        kwargs = [
            {"behavior": "test behavior 1"},
            {"behavior": "test behavior 2"},
        ]
        self.env_manager.reset(kwargs=kwargs)

        # 두 개 모두 파싱 실패하는 액션
        malformed_actions = [
            "This is not JSON at all",
            "<think>reasoning</think>still not json",
        ]
        _, _, _, infos = self.env_manager.step(malformed_actions)

        assert infos[0]["is_action_valid"] == 0
        assert infos[1]["is_action_valid"] == 0

    def test_parse_success_sets_is_action_valid_to_one(self):
        """파싱 성공 시 info['is_action_valid'] == 1."""
        kwargs = [{"behavior": "behavior A"}, {"behavior": "behavior B"}]
        self.env_manager.reset(kwargs=kwargs)

        valid_actions = [
            '{"rationale": "test", "prompt": "attack prompt 1"}',
            '{"rationale": "test", "prompt": "attack prompt 2"}',
        ]
        _, _, _, infos = self.env_manager.step(valid_actions)

        assert infos[0]["is_action_valid"] == 1
        assert infos[1]["is_action_valid"] == 1

    def test_mixed_validity_in_batch(self):
        """배치 내 valid/invalid 혼합 시 각각 올바르게 기록."""
        kwargs = [{"behavior": "beh1"}, {"behavior": "beh2"}]
        self.env_manager.reset(kwargs=kwargs)

        actions = [
            '{"rationale": "ok", "prompt": "valid attack"}',  # valid
            "completely broken text",  # invalid
        ]
        _, _, _, infos = self.env_manager.step(actions)

        assert infos[0]["is_action_valid"] == 1
        assert infos[1]["is_action_valid"] == 0

    def test_multiturn_history_accumulates_with_parse_failures(self):
        """파싱 실패해도 attacker_histories 에 raw action 이 누적됨."""
        kwargs = [{"behavior": "test behavior"}]
        self.env_manager.reset(kwargs=kwargs)

        # Turn 1: 파싱 실패
        malformed_action_t1 = "Not JSON turn 1"
        obs1, _, _, _ = self.env_manager.step([malformed_action_t1])

        # history 에 assistant 메시지로 raw action 이 추가됨
        history = self.env_manager.attacker_histories[0]
        # [system, user(init), assistant(t1), user(t1_obs)]
        assert len(history) == 4
        assert history[2]["role"] == "assistant"
        assert history[2]["content"] == malformed_action_t1

        # Turn 2: 또 파싱 실패
        malformed_action_t2 = "Also not JSON turn 2"
        obs2, _, _, _ = self.env_manager.step([malformed_action_t2])

        history = self.env_manager.attacker_histories[0]
        # [system, user(init), assistant(t1), user(t1_obs), assistant(t2), user(t2_obs)]
        assert len(history) == 6
        assert history[4]["role"] == "assistant"
        assert history[4]["content"] == malformed_action_t2

    def test_full_episode_all_parse_failures(self):
        """전체 에피소드가 모든 턴에서 파싱 실패해도 rollout 완료."""
        kwargs = [{"behavior": "harmful behavior X"}]
        self.env_manager.reset(kwargs=kwargs)

        all_infos = []
        for turn in range(3):
            malformed = f"Malformed action turn {turn}"
            _, rewards, dones, infos = self.env_manager.step([malformed])
            all_infos.append(infos[0])

            # rewards 는 mock 에서 항상 0.3
            assert rewards[0] == pytest.approx(0.3)
            # 매 턴 is_action_valid == 0
            assert infos[0]["is_action_valid"] == 0

        # 마지막 턴에서 done
        assert dones[0] == True
        # 모든 턴에서 invalid
        assert all(info["is_action_valid"] == 0 for info in all_infos)

    def test_env_step_still_receives_fallback_prompt(self):
        """파싱 실패 시 fallback prompt 가 env.step() 에 전달됨을 확인."""
        kwargs = [{"behavior": "test"}]
        self.env_manager.reset(kwargs=kwargs)

        # env.step 을 spy 해서 전달된 action 확인
        original_step = self.mock_env.step
        received_actions = []

        def spy_step(actions):
            received_actions.extend(actions)
            return original_step(actions)

        self.mock_env.step = spy_step

        # 잘린 JSON — fallback 으로 prompt 값이 추출됨
        truncated = '{"rationale": "abc", "prompt": "my fallback prompt value'
        self.env_manager.step([truncated])

        # fallback 이 env 에 전달됨
        assert len(received_actions) == 1
        assert "my fallback prompt" in received_actions[0]

    def test_observation_format_correct_after_parse_failure(self):
        """파싱 실패 후에도 observation 이 올바른 chat format (list of dicts)."""
        kwargs = [{"behavior": "test behavior"}]
        obs_init, _ = self.env_manager.reset(kwargs=kwargs)

        # init obs: list of chat messages
        assert isinstance(obs_init["text"][0], list)
        assert obs_init["text"][0][0]["role"] == "system"
        assert obs_init["text"][0][1]["role"] == "user"

        # step with parse failure
        next_obs, _, _, _ = self.env_manager.step(["not json"])

        # continuation obs: accumulated chat messages
        msgs = next_obs["text"][0]
        assert isinstance(msgs, list)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant", "user"]


# =====================================================================
# 3. apply_invalid_action_penalty 테스트
# =====================================================================
class TestApplyInvalidActionPenalty:
    """apply_invalid_action_penalty 가 파싱 실패 시 reward 를 정확히 감소."""

    @staticmethod
    def _make_data_proto(
        batch_size: int,
        prompt_len: int,
        response_len: int,
        is_action_valids: list,
        base_reward: float = 1.0,
    ):
        """DataProto 를 흉내내는 간이 구조체 생성."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        total_len = prompt_len + response_len

        class FakeDataItem:
            def __init__(self, prompts, attention_mask, is_valid):
                self.batch = {
                    "prompts": prompts,
                    "attention_mask": attention_mask,
                }
                self.non_tensor_batch = {
                    "is_action_valid": np.array([is_valid], dtype=np.float32),
                }

        class FakeDataProto:
            def __init__(self):
                self.batch = {}
                self.non_tensor_batch = {}
                self._items = []

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        data = FakeDataProto()

        # Build tensors
        prompts_all = []
        attention_masks = []
        reward_tensor = torch.zeros(batch_size, response_len)

        for i in range(batch_size):
            prompt = torch.ones(prompt_len, dtype=torch.long)
            attn = torch.ones(total_len, dtype=torch.long)
            prompts_all.append(prompt)
            attention_masks.append(attn)

            # Set reward at last response token
            reward_tensor[i, response_len - 1] = base_reward

            item = FakeDataItem(prompt, attn, is_action_valids[i])
            data._items.append(item)

        data.batch["prompts"] = torch.stack(prompts_all)
        data.batch["attention_mask"] = torch.stack(attention_masks)
        data.batch["token_level_scores"] = reward_tensor
        data.non_tensor_batch["is_action_valid"] = np.array(
            is_action_valids, dtype=np.float32
        )

        return data

    def test_penalty_applied_to_invalid_action(self):
        """valid=0 인 경우 마지막 토큰 reward 가 감소."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        data = self._make_data_proto(
            batch_size=1,
            prompt_len=10,
            response_len=20,
            is_action_valids=[0],  # invalid
            base_reward=1.0,
        )

        coef = 0.1
        data, metrics = apply_invalid_action_penalty(data, coef)

        # 마지막 토큰: 1.0 - 0.1 * 1 = 0.9
        assert data.batch["token_level_scores"][0, -1].item() == pytest.approx(0.9)
        assert metrics["episode/valid_action_ratio"] == pytest.approx(0.0)

    def test_no_penalty_for_valid_action(self):
        """valid=1 인 경우 reward 변화 없음."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        data = self._make_data_proto(
            batch_size=1,
            prompt_len=10,
            response_len=20,
            is_action_valids=[1],  # valid
            base_reward=1.0,
        )

        coef = 0.1
        data, metrics = apply_invalid_action_penalty(data, coef)

        assert data.batch["token_level_scores"][0, -1].item() == pytest.approx(1.0)
        assert metrics["episode/valid_action_ratio"] == pytest.approx(1.0)

    def test_mixed_batch_penalty(self):
        """배치 내 valid/invalid 혼합 시 invalid 만 penalty."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        data = self._make_data_proto(
            batch_size=3,
            prompt_len=10,
            response_len=20,
            is_action_valids=[1, 0, 0],  # 1 valid, 2 invalid
            base_reward=1.0,
        )

        coef = 0.5
        data, metrics = apply_invalid_action_penalty(data, coef)

        rewards = data.batch["token_level_scores"]
        assert rewards[0, -1].item() == pytest.approx(1.0)  # valid → no penalty
        assert rewards[1, -1].item() == pytest.approx(0.5)  # invalid → 1.0 - 0.5
        assert rewards[2, -1].item() == pytest.approx(0.5)  # invalid → 1.0 - 0.5
        assert metrics["episode/valid_action_ratio"] == pytest.approx(1 / 3)

    def test_penalty_with_step_rewards(self):
        """step_rewards 가 있는 경우에도 penalty 적용."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        data = self._make_data_proto(
            batch_size=2,
            prompt_len=10,
            response_len=20,
            is_action_valids=[0, 1],
            base_reward=1.0,
        )
        # step_rewards 추가
        data.batch["step_rewards"] = torch.tensor([0.5, 0.5])

        coef = 0.2
        data, metrics = apply_invalid_action_penalty(data, coef)

        # step_rewards 도 감소
        assert data.batch["step_rewards"][0].item() == pytest.approx(0.3)  # 0.5 - 0.2
        assert data.batch["step_rewards"][1].item() == pytest.approx(0.5)  # valid


# =====================================================================
# 4. 엔드-투-엔드: 파싱 실패 → env step → penalty 적용 전체 흐름
# =====================================================================
class TestEndToEndParseFailureFlow:
    """
    파싱 실패 → projection → env_manager.step → is_action_valid=0
    → apply_invalid_action_penalty 까지의 전체 흐름.
    """

    def test_e2e_parse_failure_to_penalty(self):
        """전체 파이프라인: 파싱 실패 → penalty 적용."""
        from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

        config = _make_config(max_steps=2)
        mock_env = MockMultiTurnConvEnv(batch_size=2, max_steps=2)
        env_manager = MultiTurnConvEnvironmentManager(
            envs=mock_env,
            projection_f=multiturn_conv_projection,
            config=config,
        )

        kwargs = [{"behavior": "behavior_a"}, {"behavior": "behavior_b"}]
        env_manager.reset(kwargs=kwargs)

        # Turn 1: 하나는 valid, 하나는 invalid
        actions = [
            '{"rationale": "ok", "prompt": "valid prompt"}',  # valid
            "broken text no json here",  # invalid
        ]
        _, rewards, _, infos = env_manager.step(actions)

        # validity 확인
        assert infos[0]["is_action_valid"] == 1
        assert infos[1]["is_action_valid"] == 0

        # penalty 적용 시뮬레이션
        prompt_len, response_len = 10, 20
        total_len = prompt_len + response_len

        class FakeItem:
            def __init__(self, is_valid):
                self.batch = {
                    "prompts": torch.ones(prompt_len, dtype=torch.long),
                    "attention_mask": torch.ones(total_len, dtype=torch.long),
                }
                self.non_tensor_batch = {
                    "is_action_valid": np.array(
                        [float(is_valid)], dtype=np.float32
                    ),
                }

        class FakeData:
            def __init__(self):
                self.batch = {}
                self.non_tensor_batch = {}
                self._items = []

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                return self._items[idx]

        data = FakeData()
        reward_tensor = torch.zeros(2, response_len)
        reward_tensor[0, response_len - 1] = rewards[0]
        reward_tensor[1, response_len - 1] = rewards[1]

        data.batch["prompts"] = torch.stack(
            [torch.ones(prompt_len, dtype=torch.long)] * 2
        )
        data.batch["attention_mask"] = torch.stack(
            [torch.ones(total_len, dtype=torch.long)] * 2
        )
        data.batch["token_level_scores"] = reward_tensor
        data.non_tensor_batch["is_action_valid"] = np.array(
            [infos[0]["is_action_valid"], infos[1]["is_action_valid"]],
            dtype=np.float32,
        )
        data._items = [
            FakeItem(infos[0]["is_action_valid"]),
            FakeItem(infos[1]["is_action_valid"]),
        ]

        coef = 0.1
        data, metrics = apply_invalid_action_penalty(data, coef)

        # valid action: reward 유지, invalid: reward 감소
        assert data.batch["token_level_scores"][0, -1].item() == pytest.approx(
            rewards[0]
        )
        assert data.batch["token_level_scores"][1, -1].item() == pytest.approx(
            rewards[1] - coef
        )
        assert metrics["episode/valid_action_ratio"] == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
