# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
from typing import Any, Dict, List, Optional, Tuple


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses the model output to extract JSON content.
    Handles <think>, <answer>, and markdown code blocks.
    (Same logic as attack_plugin.py)
    """
    if not text:
        return None

    # 1. Remove <think> tags and extract <answer> (if present)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    elif "</think>" in text:
        content = text.split("</think>")[-1].strip()
    else:
        content = text.strip()

    # 2. Extract JSON from Markdown block
    match = re.search(r"```json\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 3. Iterative JSON finding (Brute-force)
    decoder = json.JSONDecoder()
    pos = len(content)
    while True:
        start = content.rfind("{", 0, pos)
        if start == -1:
            break
        try:
            obj, _ = decoder.raw_decode(content[start:])
            return obj
        except json.JSONDecodeError:
            pos = start

    return None


def multiturn_conv_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Project attacker LLM outputs into parsed prompt strings.

    For each action:
      1. Parse JSON (handling <think> tags, markdown, etc.)
      2. Extract the "prompt" field

    Returns:
        results: List[str] — extracted prompt strings
        valids: List[int] — 1 if valid parse, 0 otherwise
    """
    results: List[str] = []
    valids: List[int] = []

    for action in actions:
        parsed = parse_json_response(action)
        if parsed and "prompt" in parsed:
            results.append(parsed["prompt"])
            valids.append(1)
        else:
            results.append("")
            valids.append(0)

    return results, valids


# Keep the original search_projection for backward compatibility
def _postprocess_action(action: str) -> str:
    """Trim everything *after* the first closing </search> or </answer> tag."""
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Original search projection — kept for backward compatibility."""
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    re_search_block = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
    re_answer_block = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_search_tag = re.compile(r"<search>", re.IGNORECASE)
    re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)

    for i, action in enumerate(actions):
        original_action = action
        trimmed_action = _postprocess_action(action)

        m = re_search_block.search(trimmed_action)
        if m:
            results.append(f"<search>{m.group(1).strip()}</search>")
        else:
            m = re_answer_block.search(trimmed_action)
            if m:
                results.append(f"<answer>{m.group(1).strip()}</answer>")
            else:
                results.append("")
                valids[i] = 0

        n_search = len(re_search_tag.findall(original_action))
        n_answer = len(re_answer_tag.findall(original_action))

        if n_search and n_answer:
            valids[i] = 0
            continue
        if n_search > 1 or n_answer > 1:
            valids[i] = 0

    return results, valids
