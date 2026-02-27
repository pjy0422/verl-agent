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


def _extract_prompt_value_raw(text: str) -> Optional[str]:
    """
    Regex-extract the value after a "prompt" key from malformed JSON or markdown.

    Handles:
      JSON malformed:  {"rationale": "abc", "prompt": ewmviwemiv}
      JSON truncated:  {"rationale": "abc", "prompt": "some text that got cut off...
      Markdown bold:   **prompt**: actual attack text here
      Markdown header: ## Prompt\nthe content
    """
    # --- JSON-style: "prompt" : value ---
    m = re.search(r'"prompt"\s*:\s*', text)
    if m:
        rest = text[m.end():]

        # Case 1: quoted value (possibly unterminated)
        if rest.startswith('"'):
            qm = re.match(r'"((?:[^"\\]|\\.)*)"?', rest)
            if qm and qm.group(1).strip():
                return qm.group(1).strip()

        # Case 2: unquoted value — grab until }, comma, or end of string
        um = re.match(r'([^,}\]]+)', rest)
        if um and um.group(1).strip():
            return um.group(1).strip()

    # --- Markdown-style: **prompt**: value  /  **Prompt:** value ---
    md = re.search(
        r'(?:\*\*|#+\s*)prompt\s*(?:\*\*)?\s*:?\s*(?:\*\*)?\s*(.+)',
        text,
        re.IGNORECASE,
    )
    if md:
        # Value is everything on the same line (and continuation lines until next key)
        start = md.end(1)
        value = md.group(1).strip()
        # If multiline, grab until next markdown key or end
        rest_text = text[md.start(1):]
        # Stop at next **key**: or ## header or end
        block = re.split(r'\n\s*(?:\*\*\w|\#{1,3}\s|\-\s\*\*)', rest_text, maxsplit=1)[0]
        block = block.strip()
        if len(block) > len(value):
            value = block
        if value:
            return value

    return None


def _extract_fallback_prompt(action: str, parsed: Optional[Dict]) -> str:
    """
    Best-effort prompt extraction when normal JSON parsing fails.

    Priority:
      1. Regex-extract value after "prompt" key from raw text (handles malformed JSON)
      2. JSON parsed but no "prompt" → use longest string value
      3. No JSON → strip <think> tags and use raw text
      4. Absolute fallback → generic continuation
    """
    # Strip <think> tags first for all fallback paths
    content = action or ""
    answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    elif "</think>" in content:
        content = content.split("</think>")[-1].strip()

    # Fallback 1: Regex-extract "prompt" value from raw (malformed) JSON
    raw_prompt = _extract_prompt_value_raw(content)
    if raw_prompt and len(raw_prompt) > 5:
        return raw_prompt

    # Fallback 2: JSON parsed but missing "prompt" key → use other fields
    if parsed:
        for key in ("attack_prompt", "message", "content", "query", "input"):
            if key in parsed and isinstance(parsed[key], str) and parsed[key].strip():
                return parsed[key].strip()
        string_vals = [
            v for v in parsed.values()
            if isinstance(v, str) and len(v.strip()) > 20
        ]
        if string_vals:
            return max(string_vals, key=len).strip()

    # Fallback 3: Use raw text as-is
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```", "", content)
    content = content.strip()
    if content.startswith("{"):
        # Extract readable text between quotes from broken JSON
        quoted = re.findall(r'"([^"]{20,})"', content)
        if quoted:
            content = max(quoted, key=len)

    content = content.strip()
    if len(content) > 10:
        return content

    # Fallback 4: absolute last resort
    return "Continue the conversation."


def multiturn_conv_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Project attacker LLM outputs into parsed prompt strings.

    For each action:
      1. Parse JSON → extract "prompt" field (valid=1)
      2. If that fails → best-effort fallback extraction (valid=0)

    Returns:
        results: List[str] — extracted prompt strings (always non-empty)
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
            # Fallback: salvage whatever we can
            results.append(_extract_fallback_prompt(action, parsed))
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
