"""Unit tests for XML prompt format support."""

import pytest
from agent_system.environments.env_package.multiturn_conv.projection import (
    parse_xml_response,
    validate_xml_format,
    multiturn_conv_projection_xml,
)
from agent_system.environments.prompts.multiturn import (
    MULTITURN_SYSTEM_PROMPT,
    MULTITURN_SYSTEM_PROMPT_XML,
    get_system_prompt,
)


# ============================================================
# Tests for parse_xml_response
# ============================================================
class TestParseXmlResponse:
    def test_well_formed(self):
        text = "<reasoning>reasoning here</reasoning><answer>attack prompt here</answer>"
        result = parse_xml_response(text)
        assert result is not None
        assert result["rationale"] == "reasoning here"
        assert result["prompt"] == "attack prompt here"

    def test_only_answer(self):
        text = "<answer>just the prompt</answer>"
        result = parse_xml_response(text)
        assert result is not None
        assert result["prompt"] == "just the prompt"
        assert "rationale" not in result

    def test_empty_tags(self):
        text = "<reasoning></reasoning><answer></answer>"
        result = parse_xml_response(text)
        assert result is not None
        assert result["prompt"] == ""

    def test_no_tags(self):
        text = "Just some random text without any XML tags"
        result = parse_xml_response(text)
        assert result is None

    def test_empty_input(self):
        assert parse_xml_response("") is None
        assert parse_xml_response(None) is None

    def test_multiline_content(self):
        text = "<reasoning>\nline1\nline2\n</reasoning><answer>\nprompt\nline2\n</answer>"
        result = parse_xml_response(text)
        assert result is not None
        assert "line1" in result["rationale"]
        assert "prompt" in result["prompt"]

    def test_mixed_with_other_text(self):
        text = "Some prefix <reasoning>reasoning</reasoning> middle <answer>prompt</answer> suffix"
        result = parse_xml_response(text)
        assert result is not None
        assert result["rationale"] == "reasoning"
        assert result["prompt"] == "prompt"

    def test_nested_tags_graceful(self):
        text = "<reasoning>outer <reasoning>inner</reasoning> end</reasoning><answer>result</answer>"
        result = parse_xml_response(text)
        assert result is not None
        # re.search with DOTALL and non-greedy will match the first <reasoning>...</reasoning>
        assert result["prompt"] == "result"


# ============================================================
# Tests for validate_xml_format
# ============================================================
class TestValidateXmlFormat:
    def test_both_present_nonempty(self):
        text = "<reasoning>reasoning</reasoning><answer>prompt</answer>"
        assert validate_xml_format(text) == 1.0

    def test_only_answer_nonempty(self):
        text = "<answer>prompt only</answer>"
        assert validate_xml_format(text) == 0.5

    def test_no_tags(self):
        assert validate_xml_format("no tags here") == 0.0

    def test_empty_string(self):
        assert validate_xml_format("") == 0.0

    def test_empty_content_inside_tags(self):
        text = "<reasoning></reasoning><answer></answer>"
        assert validate_xml_format(text) == 0.0

    def test_think_only_no_answer(self):
        text = "<reasoning>reasoning only</reasoning>"
        assert validate_xml_format(text) == 0.0

    def test_answer_with_whitespace_only(self):
        text = "<reasoning>reasoning</reasoning><answer>   </answer>"
        assert validate_xml_format(text) == 0.0

    def test_answer_nonempty_think_empty(self):
        text = "<reasoning>  </reasoning><answer>valid prompt</answer>"
        assert validate_xml_format(text) == 0.5


# ============================================================
# Tests for multiturn_conv_projection_xml
# ============================================================
class TestMultiturnConvProjectionXml:
    def test_valid_xml(self):
        actions = ["<reasoning>reasoning</reasoning><answer>attack prompt</answer>"]
        results, valids, scores = multiturn_conv_projection_xml(actions)
        assert results == ["attack prompt"]
        assert valids == [1]
        assert scores == [1.0]

    def test_fallback_case(self):
        actions = ["Just some text without XML tags but long enough to be used"]
        results, valids, scores = multiturn_conv_projection_xml(actions)
        assert valids == [0]
        assert scores == [0.0]
        assert len(results[0]) > 0  # fallback should extract something

    def test_batch_mixed(self):
        actions = [
            "<reasoning>r1</reasoning><answer>prompt1</answer>",
            "no xml here but long enough text for fallback extraction",
            "<answer>only answer</answer>",
        ]
        results, valids, scores = multiturn_conv_projection_xml(actions)
        assert len(results) == 3
        assert valids == [1, 0, 1]
        assert scores == [1.0, 0.0, 0.5]
        assert results[0] == "prompt1"
        assert results[2] == "only answer"

    def test_empty_answer_tag(self):
        actions = ["<reasoning>reasoning</reasoning><answer></answer>"]
        results, valids, scores = multiturn_conv_projection_xml(actions)
        assert valids == [0]  # empty prompt -> fallback
        assert scores == [0.0]  # empty content -> 0.0


# ============================================================
# Tests for get_system_prompt
# ============================================================
class TestGetSystemPrompt:
    def test_json_returns_json_prompt(self):
        prompt = get_system_prompt("json")
        assert prompt is MULTITURN_SYSTEM_PROMPT
        assert "JSON" in prompt

    def test_xml_returns_xml_prompt(self):
        prompt = get_system_prompt("xml")
        assert prompt is MULTITURN_SYSTEM_PROMPT_XML
        assert "XML" in prompt
        assert "<reasoning>" in prompt
        assert "<answer>" in prompt

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            get_system_prompt("yaml")

    def test_xml_prompt_has_behavior_placeholder(self):
        prompt = get_system_prompt("xml")
        assert "{behavior}" in prompt
        # Should be formattable
        formatted = prompt.format(behavior="test behavior")
        assert "test behavior" in formatted
